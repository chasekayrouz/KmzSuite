#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <zlib.h>

/* Faithful acceleration of kmzconvert_core.binary (scan_metadata, decode_tokens,
 * measure_spectrum, frame_profile), mass_correction.MassCorrection.correct,
 * kmzcore.sparse_profile (source_sampling_check v3, rescue_selection), zlib and
 * base64; see those modules and THIRD_PARTY.md for the codec's provenance.
 * Results must equal the Python/NumPy references bit for bit, so floating-point
 * contraction stays disabled. Every GIL-free loop reads only immutable bytes
 * arguments (or private copies) and writes only private allocations. */
#pragma STDC FP_CONTRACT OFF
_Static_assert(sizeof(int64_t) == 8 && sizeof(double) == 8,
               "The native codec requires 64-bit integer and double arrays");

/* Release the GIL only for work that outweighs the thread hand-off, as NumPy does. */
#define RELEASE(items) PyThreadState *released = (items) > 4096 ? PyEval_SaveThread() : NULL
#define REACQUIRE() if (released) PyEval_RestoreThread(released)

/* No Python API calls in this loop: the caller owns immutable input bytes and
 * private output buffers for the entire (possibly) GIL-free interval. */
static const char *decode(const unsigned char *b, Py_ssize_t end,
                         char *x, char *y, Py_ssize_t *count, int *overflow)
{
    if (end < 11 || memcmp(b, "\xff\xff\xff\xff", 4) || b[8])
        return "Unsupported X500 spectrum block header";
    uint64_t coordinate = 0;
    for (unsigned j = 0; j < 4; j++) coordinate |= (uint64_t)b[4+j] << (8*j);
    if (coordinate % 4) return "Unaligned TOF seed";
    Py_ssize_t pos = 9, used = 0;
    int terminated = 0;
    while (pos < end) {
        if (end-pos <= 4) {
            Py_ssize_t j = pos;
            while (j < end && b[j] == 255) j++;
            if (j == end) { terminated = 1; break; }
        }
        unsigned c = b[pos++];
        uint64_t delta = 1, value;
        if (c >= 128) {
            if (c >= 252 && c <= 254) {
                unsigned n = c == 252 ? 1 : c == 253 ? 2 : 4;
                if (end-pos <= n) return "Truncated TOF delta";
                delta = 0;
                for (unsigned j = 0; j < n; j++) delta |= (uint64_t)b[pos++] << (8*j);
                delta++;
            } else {
                if (c == 255) return "Unexpected 0xff delta outside terminator";
                delta = c-127;
            }
            if (pos >= end) return "Missing intensity";
            c = b[pos++];
        }
        value = c;
        if (c >= 124 && c <= 126) {
            unsigned n = c == 124 ? 1 : c == 125 ? 2 : 4;
            if (end-pos < n) return "Truncated intensity";
            value = 0;
            for (unsigned j = 0; j < n; j++) value |= (uint64_t)b[pos++] << (8*j);
        } else if (c == 127) return "Unsupported intensity prefix 0x7f";
        if (coordinate > INT64_MAX-delta*4) {
            *overflow = 1;
            return "TOF coordinate exceeds int64";
        }
        coordinate += delta*4;
        int64_t coord = (int64_t)coordinate;
        double intensity = (double)value;
        /* memcpy avoids assumptions about Python byte-buffer alignment. */
        memcpy(x+used*8, &coord, 8);
        memcpy(y+used*8, &intensity, 8);
        used++;
    }
    if (!terminated || !used) return "Missing spectrum terminator or empty spectrum";
    *count = used;
    return NULL;
}

static PyObject *decode_tokens(PyObject *self, PyObject *arg)
{
    (void)self;
    if (!PyBytes_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "decode_tokens requires immutable bytes");
        return NULL;
    }
    Py_ssize_t size = PyBytes_GET_SIZE(arg);
    /* At most one peak per body byte. Guard multiplication before allocation. */
    Py_ssize_t capacity = size > 9 ? size-9 : 0;
    if (capacity > PY_SSIZE_T_MAX/8) return PyErr_NoMemory();
    PyObject *x = PyBytes_FromStringAndSize(NULL, capacity*8);
    if (!x) return NULL;
    PyObject *y = PyBytes_FromStringAndSize(NULL, capacity*8);
    if (!y) { Py_DECREF(x); return NULL; }
    const unsigned char *input = (const unsigned char *)PyBytes_AS_STRING(arg);
    char *xp = PyBytes_AS_STRING(x), *yp = PyBytes_AS_STRING(y);
    Py_ssize_t count = 0;
    int overflow = 0;
    const char *error;
    RELEASE(size);
    error = decode(input, size, xp, yp, &count, &overflow);
    REACQUIRE();
    if (error) {
        Py_DECREF(x); Py_DECREF(y);
        PyErr_SetString(overflow ? PyExc_OverflowError : PyExc_ValueError, error);
        return NULL;
    }
    /* Buffers have never escaped and each has exactly one owner. */
    if (_PyBytes_Resize(&x, count*8) < 0) { Py_DECREF(y); return NULL; }
    if (_PyBytes_Resize(&y, count*8) < 0) { Py_DECREF(x); return NULL; }
    return Py_BuildValue("NN", x, y);
}

/* binary.varint: 0 where the reference raises (truncated, or >= 2**64). */
static int varint(const unsigned char *d, Py_ssize_t *pos, Py_ssize_t end, uint64_t *value)
{
    uint64_t result = 0;
    for (int shift = 0; shift < 70; shift += 7) {
        if (*pos >= end) return 0;
        unsigned b = d[(*pos)++];
        if (shift == 63 && (b & 127) > 1) return 0;
        result |= (uint64_t)(b & 127) << shift;
        if (b < 128) { *value = result; return 1; }
    }
    return 0;
}

/* One length-delimited field with the expected tag; start and stop bound its bytes. */
static int field(const unsigned char *d, Py_ssize_t *pos, Py_ssize_t end, uint64_t tag,
                 Py_ssize_t *start, Py_ssize_t *stop)
{
    uint64_t found, size;
    if (!varint(d, pos, end, &found) || found != tag || !varint(d, pos, end, &size)
            || size > (uint64_t)(end - *pos)) return 0;
    *start = *pos;
    *pos += (Py_ssize_t)size;
    *stop = *pos;
    return 1;
}

/* binary.scan_metadata for its qualified layout only: (a, b, [(x, y), ...]), or
 * None for any other input, which the reference then parses or rejects. */
static PyObject *scan_metadata(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data, *checkpoints = NULL, *result = NULL;
    Py_ssize_t start, end;
    if (!PyArg_ParseTuple(args, "Onn", &data, &start, &end)) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(data, &view, PyBUF_SIMPLE) < 0) return NULL;
    const unsigned char *d = view.buf;
    Py_ssize_t pos = start, cal, cal_end, body;
    uint64_t size, tag;
    double ab[2];
    if (start < 0 || start > end || end > view.len || !varint(d, &pos, end, &size)
            || size != (uint64_t)(end - pos) || !field(d, &pos, end, 0x0a, &cal, &cal_end))
        goto other;
    body = cal;
    for (int i = 0; i < 2; i++) {
        if (!varint(d, &body, cal_end, &tag) || tag != (uint64_t)(i ? 0x11 : 0x09) || cal_end - body < 8) goto other;
        uint64_t bits = 0;
        for (int j = 0; j < 8; j++) bits |= (uint64_t)d[body+j] << (8*j);
        memcpy(ab+i, &bits, 8);
        body += 8;
    }
    if (body != cal_end || !(checkpoints = PyList_New(0))) goto other;
    while (pos < end) {
        Py_ssize_t pair, pair_end;
        uint64_t x, y;
        if (!field(d, &pos, end, 0x12, &pair, &pair_end) || !varint(d, &pair, pair_end, &tag) || tag != 0x08
                || !varint(d, &pair, pair_end, &x) || !varint(d, &pair, pair_end, &tag) || tag != 0x10
                || !varint(d, &pair, pair_end, &y) || pair != pair_end) goto other;
        PyObject *item = Py_BuildValue("(KK)", (unsigned long long)x, (unsigned long long)y);
        if (!item || PyList_Append(checkpoints, item) < 0) { Py_XDECREF(item); goto failed; }
        Py_DECREF(item);
    }
    result = Py_BuildValue("ddO", ab[0], ab[1], checkpoints);
    goto failed;
other:
    if (!PyErr_Occurred()) result = Py_NewRef(Py_None);
failed:
    Py_XDECREF(checkpoints);
    PyBuffer_Release(&view);
    return result;
}

/* np.unique(np.concatenate((x-4, x, x+4))) for strictly increasing x whose gaps
 * are multiples of 4, each x's intensity at its own grid point and +0.0 elsewhere,
 * calibrated as np.square(a * (grid * tdc - b)). A gap of 4 makes x the previous
 * x+4 (slot p-1); a gap of 8 makes x-4 that slot. Writes are branch-free: the
 * provisional slots are overwritten. The grid is staged in mz's storage. */
static Py_ssize_t frame(const int64_t *x, const double *y, Py_ssize_t n,
                        double a, double b, double tdc, double *mz, double *out)
{
    Py_ssize_t p = 0;
    for (Py_ssize_t j = 0; j < n; j++) {
        uint64_t gap = j ? (uint64_t)x[j] - (uint64_t)x[j-1] : 12;
        int64_t grid[3] = {x[j]-4, x[j], x[j]+4};
        memcpy(mz+p, grid, 8); out[p] = 0.0;
        p += gap > 8;
        p -= gap == 4;
        memcpy(mz+p, grid+1, 8); out[p] = y[j];
        p++;
        memcpy(mz+p, grid+2, 8); out[p] = 0.0;
        p++;
    }
    for (Py_ssize_t i = 0; i < p; i++) {
        int64_t g;
        memcpy(&g, mz+i, 8);
        double t = a * ((double)g * tdc - b);
        mz[i] = t * t;
    }
    return p;
}

/* binary.measure_spectrum: calibrated m/z (np.square(a * (x * tdc - b))) and the
 * reader's measured-spectrum validity. Positive integer intensities summing below
 * 2**53 make every partial sum exact, so the integer total and maximum equal
 * NumPy's float sum/max in any order; otherwise *exact is 0 and NumPy decides
 * (including the sign of a zero maximum). */
static void measure(const int64_t *x, const double *y, Py_ssize_t n, double a, double b, double tdc,
                    double *mz, double *total, double *top, int *exact, int *valid)
{
    const double limit = 9007199254740992.0;  /* 2**53 */
    uint64_t sum = 0;
    double high = 0.0;
    int integral = 1, finite = 1, positive = 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        double t = a * ((double)x[i] * tdc - b);
        mz[i] = t * t;
        finite &= isfinite(mz[i]) != 0;
        positive &= y[i] > 0;
        if (integral && y[i] > 0 && y[i] < limit && y[i] == floor(y[i])) {
            sum += (uint64_t)y[i];
            integral = sum < (uint64_t)1 << 53;
            if (y[i] > high) high = y[i];
        } else integral = 0;
    }
    int increasing = 1;
    for (Py_ssize_t i = 1; finite && increasing && i < n; i++) increasing = mz[i] - mz[i-1] > 0;
    *exact = integral;
    *total = (double)sum;
    *top = high;
    *valid = finite && increasing && mz[0] > 0 && positive;
}

typedef struct { size_t size, used; uint64_t *keys; Py_ssize_t *counts; } Table;

static size_t slot(const Table *t, uint64_t key)
{
    size_t h = (size_t)(key * 0x9e3779b97f4a7c15ULL >> 40) & (t->size-1);
    while (t->counts[h] && t->keys[h] != key) h = (h+1) & (t->size-1);
    return h;
}

static int resize(Table *t, size_t size)
{
    Table old = *t;
    t->size = size;
    t->keys = PyMem_RawMalloc(size * sizeof(uint64_t));
    t->counts = PyMem_RawCalloc(size, sizeof(Py_ssize_t));
    int failed = !t->keys || !t->counts;
    for (size_t j = 0; !failed && j < old.size; j++)
        if (old.counts[j]) {
            size_t h = slot(t, old.keys[j]);
            t->keys[h] = old.keys[j];
            t->counts[h] = old.counts[j];
        }
    PyMem_RawFree(old.keys); PyMem_RawFree(old.counts);
    return failed ? -1 : 0;
}

/* Sparse Profile v3 event unit: most frequent positive intensity, smallest on
 * ties (np.unique order). Integer counts below SMALL are tallied branch-free (slot
 * 0 collects everything else); other positive finite doubles are equal iff their
 * bits are and use an open-addressing table that starts small. */
static int event_unit(const double *y, Py_ssize_t n, double *unit)
{
    enum { SMALL = 4096 };
    Table t = {0, 0, NULL, NULL};
    Py_ssize_t best = 0, *small = PyMem_RawCalloc(SMALL, sizeof(Py_ssize_t));
    int failed = !small || resize(&t, 1024);
    *unit = 0.0;
    for (Py_ssize_t i = 0; !failed && i < n; i++) {
        double v = y[i];
        int k = v > 0 && v < SMALL ? (int)v : 0;
        k = v == (double)k ? k : 0;
        small[k]++;
        if (k || !(v > 0)) continue;
        uint64_t key;
        memcpy(&key, &v, 8);
        size_t h = slot(&t, key);
        if (!t.counts[h]) {
            if (2 * (t.used+1) > t.size) {
                if ((failed = resize(&t, 2 * t.size))) break;
                h = slot(&t, key);
            }
            t.keys[h] = key;
            t.used++;
        }
        t.counts[h]++;
    }
    for (int k = 1; !failed && k < SMALL; k++)
        if (small[k] > best) { best = small[k]; *unit = k; }
    for (size_t h = 0; !failed && h < t.size; h++) {
        double v;
        memcpy(&v, t.keys + h, 8);
        if (t.counts[h] > best || (t.counts[h] && t.counts[h] == best && v < *unit)) { best = t.counts[h]; *unit = v; }
    }
    PyMem_RawFree(small); PyMem_RawFree(t.keys); PyMem_RawFree(t.counts);
    return failed;
}

/* kmzcore.sparse_profile.validate_arrays value checks for aligned float64 arrays. */
static const char *invalid(const double *mz, const double *y, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++)
        if (!isfinite(mz[i])) return "Nonfinite measurements";
    for (Py_ssize_t i = 0; i < n; i++)
        if (!isfinite(y[i])) return "Nonfinite measurements";
    if (n && mz[0] <= 0) return "Expected positive, strictly increasing m/z";
    for (Py_ssize_t i = 1; i < n; i++)
        if (mz[i] - mz[i-1] <= 0) return "Expected positive, strictly increasing m/z";
    return NULL;
}

/* kmzcore.sparse_profile.candidate_regions after validation: maximal runs >2u
 * with at least three points and apex >8u; a run's maximum is its apex. */
static Py_ssize_t regions(const double *mz, const double *y, Py_ssize_t n, double unit,
                          int64_t *starts, int64_t *ends, double *apex, double *lower, double *upper)
{
    Py_ssize_t used = 0;
    if (!(unit > 0)) return 0;
    double above = 2.0 * unit, rescue = 8.0 * unit;
    for (Py_ssize_t i = 0; i < n;) {
        if (!(y[i] > above)) { i++; continue; }
        Py_ssize_t start = i;
        double top = y[i];
        while (++i < n && y[i] > above) if (y[i] > top) top = y[i];
        if (i - start >= 3 && top > rescue) {
            starts[used] = start; ends[used] = i; apex[used] = top;
            lower[used] = mz[start]; upper[used] = mz[i-1];
            used++;
        }
    }
    return used;
}

static Py_ssize_t root(Py_ssize_t *parent, Py_ssize_t i)
{
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
}

/* kmzcore.sparse_profile.rescue_selection for spectra whose candidate spans are
 * finite and ordered: closed spans of consecutive spectra link where they overlap
 * (np.searchsorted(b.upper, a.lower, "left") up to np.searchsorted(b.lower,
 * a.upper, "right")), and a candidate is chosen when its component holds a seed. */
static int rescue(const int64_t *sizes, Py_ssize_t spectra, const double *lower, const double *upper,
                  const char *seeds, Py_ssize_t total, char *chosen)
{
    Py_ssize_t *parent = PyMem_RawMalloc((total ? total : 1) * sizeof(Py_ssize_t));
    char *seeded = PyMem_RawCalloc(total ? total : 1, 1);
    if (!parent || !seeded) { PyMem_RawFree(parent); PyMem_RawFree(seeded); return -1; }
    for (Py_ssize_t i = 0; i < total; i++) parent[i] = i;
    Py_ssize_t a = 0;
    for (Py_ssize_t j = 0; j + 1 < spectra; j++) {
        Py_ssize_t b = a + sizes[j], end = b + sizes[j+1];
        for (Py_ssize_t i = a; i < b; i++) {
            Py_ssize_t lo = b, hi = end;
            while (lo < hi) { Py_ssize_t mid = lo + (hi-lo)/2; if (upper[mid] < lower[i]) lo = mid+1; else hi = mid; }
            Py_ssize_t first = lo;
            lo = b, hi = end;
            while (lo < hi) { Py_ssize_t mid = lo + (hi-lo)/2; if (lower[mid] <= upper[i]) lo = mid+1; else hi = mid; }
            for (Py_ssize_t k = first; k < lo; k++) {
                Py_ssize_t x = root(parent, i), y = root(parent, k);
                if (x != y) parent[x > y ? x : y] = x > y ? y : x;
            }
        }
        a = b;
    }
    for (Py_ssize_t i = 0; i < total; i++) if (seeds[i]) seeded[root(parent, i)] = 1;
    for (Py_ssize_t i = 0; i < total; i++) chosen[i] = seeded[root(parent, i)];
    PyMem_RawFree(parent); PyMem_RawFree(seeded);
    return 0;
}

/* rescue() of native int64 sizes and float64 lower/upper spans (immutable bytes,
 * copied to aligned private storage) and 0/1 seed bytes, else None. */
static PyObject *rescue_selection(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *so, *lo, *uo, *seeds, *result = NULL, *chosen = NULL;
    if (!PyArg_ParseTuple(args, "SSSS", &so, &lo, &uo, &seeds)) return NULL;
    Py_ssize_t spectra = PyBytes_GET_SIZE(so) / 8, total = PyBytes_GET_SIZE(lo) / 8, counted = 0, pos = 0;
    if (PyBytes_GET_SIZE(so) % 8 || PyBytes_GET_SIZE(lo) % 8 || PyBytes_GET_SIZE(uo) != 8*total
            || PyBytes_GET_SIZE(seeds) != total) {
        PyErr_SetString(PyExc_ValueError, "Inconsistent rescue selection arrays");
        return NULL;
    }
    char *copies = PyMem_RawMalloc(8*(spectra + 2*total) + 1);
    if (!copies) return PyErr_NoMemory();
    memcpy(copies, PyBytes_AS_STRING(so), 8*spectra);
    memcpy(copies + 8*spectra, PyBytes_AS_STRING(lo), 8*total);
    memcpy(copies + 8*(spectra + total), PyBytes_AS_STRING(uo), 8*total);
    const int64_t *sizes = (const int64_t *)copies;
    const double *lower = (const double *)(copies + 8*spectra), *upper = lower + total;
    int simple = 1;
    for (Py_ssize_t j = 0; simple && j < spectra; j++) {
        simple = sizes[j] >= 0 && sizes[j] <= total - counted;
        for (Py_ssize_t i = pos; simple && i < pos + sizes[j]; i++)
            simple = isfinite(lower[i]) && isfinite(upper[i]) && (i == pos || (lower[i-1] <= lower[i] && upper[i-1] <= upper[i]));
        counted += simple ? sizes[j] : 0;
        pos = counted;
    }
    /* Outside this precondition the caller keeps the reference selection. */
    if (!simple || counted != total) { result = Py_NewRef(Py_None); goto done; }
    if (!(chosen = PyByteArray_FromStringAndSize(NULL, total))) goto done;
    if (rescue(sizes, spectra, lower, upper, PyBytes_AS_STRING(seeds), total, PyByteArray_AS_STRING(chosen)) < 0) {
        PyErr_NoMemory();
        goto done;
    }
    result = Py_NewRef(chosen);
done:
    Py_XDECREF(chosen);
    PyMem_RawFree(copies);
    return result;
}

/* A finite, positive, strictly increasing axis (np.diff(axis) > 0). */
static int axis(const double *v, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++)
        if (!isfinite(v[i]) || v[i] <= 0 || (i && !(v[i] - v[i-1] > 0))) return 0;
    return 1;
}

/* Standard padded base64 of n bytes into (n+2)/3*4 output bytes. */
static void base64(const unsigned char *in, Py_ssize_t n, unsigned char *out)
{
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    Py_ssize_t i = 0;
    for (; i + 3 <= n; i += 3, out += 4) {
        uint32_t v = (uint32_t)in[i] << 16 | (uint32_t)in[i+1] << 8 | in[i+2];
        out[0] = table[v >> 18]; out[1] = table[v >> 12 & 63]; out[2] = table[v >> 6 & 63]; out[3] = table[v & 63];
    }
    if (i < n) {
        uint32_t v = (uint32_t)in[i] << 16 | (i+1 < n ? (uint32_t)in[i+1] << 8 : 0);
        out[0] = table[v >> 18]; out[1] = table[v >> 12 & 63];
        out[2] = i+1 < n ? table[v >> 6 & 63] : '=';
        out[3] = '=';
    }
}

/* One side's supported lower-quartile estimate from its ascending values. */
static double estimate(const double *ordered, int count)
{
    if (count < 3) return 0.0;
    double rank = .25 * (double)(count-1), lo = floor(rank), hi = ceil(rank);
    double a = ordered[(int)lo], b = ordered[(int)hi];
    double result = a + (b - a) * (rank - lo);
    double low = .8 * result, high = 1.2 * result;
    int near = 0;
    for (int i = 0; i < count; i++) near += ordered[i] >= low && ordered[i] <= high;
    return near >= 2 ? result : 0.0;
}

static double partial(const double *values, int count)
{
    double ordered[8];
    for (int i = 0; i < count; i++) {
        int j = i;
        while (j > 0 && ordered[j-1] > values[i]) { ordered[j] = ordered[j-1]; j--; }
        ordered[j] = values[i];
    }
    return estimate(ordered, count);
}

/* estimate() for every complete eight-value window: rank 1.75 interpolates the
 * second and third smallest values (rank - lo == .75 exactly). Deltas are finite
 * and non-negative (never -0.0), so branchless min/max select the same values. */
static void full_windows(const double *delta, Py_ssize_t windows, double *full)
{
    for (Py_ssize_t s = 0; s < windows; s++) {
        const double *w = delta + s;
        double a = INFINITY, b = INFINITY, c = INFINITY;
        for (int j = 0; j < 8; j++) {
            c = fmin(c, fmax(b, w[j]));
            b = fmin(b, fmax(a, w[j]));
            a = fmin(a, w[j]);
        }
        double result = b + (c - b) * .75, low = .8 * result, high = 1.2 * result;
        int near = 0;
        for (int j = 0; j < 8; j++) near += (w[j] >= low) & (w[j] <= high);
        full[s] = near >= 2 ? result : 0.0;
    }
}

/* Row i's sides are delta[i-8:i] and delta[i+1:i+9]; complete windows are shared
 * (left of i+9 == right of i). */
static void gaps(const double *mz, Py_ssize_t n, double *delta, double *full, char *out)
{
    enum { R = 8 };
    Py_ssize_t m = n-1;
    double previous = sqrt(mz[0]);
    for (Py_ssize_t i = 0; i < m; i++) {
        double next = sqrt(mz[i+1]);
        delta[i] = next - previous;
        previous = next;
    }
    if (m >= R) full_windows(delta, m-R+1, full);
    for (Py_ssize_t i = 0; i < m; i++) {
        double left = i >= R ? full[i-R] : partial(delta, (int)i);
        double right = i+R < m ? full[i+1] : partial(delta+i+1, (int)(m-1-i));
        double local = left >= right ? left : right;
        out[i] = local == 0 || delta[i] <= 4.0 * local;
    }
}

/* A private, writable buffer allocated without the GIL; NumPy views keep it alive
 * and behave like freshly allocated arrays. */
typedef struct { PyObject_HEAD char *data; Py_ssize_t size; } Buffer;

static void buffer_dealloc(Buffer *self)
{
    PyMem_RawFree(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int buffer_get(Buffer *self, Py_buffer *view, int flags)
{
    return PyBuffer_FillInfo(view, (PyObject *)self, self->data, self->size, 0, flags);
}

static Py_ssize_t buffer_length(Buffer *self)
{
    return self->size;
}

static PyBufferProcs buffer_procs = {(getbufferproc)buffer_get, NULL};
static PyMappingMethods buffer_mapping = {(lenfunc)buffer_length, NULL, NULL};

static PyTypeObject BufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "kmzconvert_core._native.Buffer",
    .tp_basicsize = sizeof(Buffer),
    .tp_dealloc = (destructor)buffer_dealloc,
    .tp_as_buffer = &buffer_procs,
    .tp_as_mapping = &buffer_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Owned native array storage.",
};

/* Takes ownership of data (freed on failure). */
static PyObject *adopt(char *data, Py_ssize_t size)
{
    Buffer *buffer = PyObject_New(Buffer, &BufferType);
    if (!buffer) { PyMem_RawFree(data); return NULL; }
    buffer->data = data;
    buffer->size = size;
    return (PyObject *)buffer;
}

/* frame()'s point count: three per peak, less those shared with the previous one. */
static Py_ssize_t framed_size(const int64_t *x, Py_ssize_t n)
{
    Py_ssize_t p = 3;
    for (Py_ssize_t j = 1; j < n; j++) {
        uint64_t gap = (uint64_t)x[j] - (uint64_t)x[j-1];
        p += 2 + (gap > 8) - (gap == 4);
    }
    return p;
}

/* One spectrum through decode_tokens, measure_spectrum, frame_profile and, for a
 * nonzero denominator, MassCorrection.correct. complete: the framed arrays hold the
 * reference values (status 1|2 marks an invalid original|corrected axis). Otherwise
 * an exception applies (error, failed), the reader's guards reject the spectrum
 * (!valid) or NumPy decides (!exact, !simple). No Python API: runs without the GIL. */
typedef struct {
    double total, top, first, last;
    int exact, valid, simple, complete, status, overflow, failed;
    const char *error;
    Py_ssize_t used;
    double *mz, *y;  /* PyMem_Raw allocations owned by the caller */
} Profile;

static void profile_block(const unsigned char *input, Py_ssize_t size, double a, double b, double tdc,
                          double denominator, Profile *p)
{
    memset(p, 0, sizeof *p);
    Py_ssize_t capacity = size > 9 ? size-9 : 0, count = 0;
    char *raw = PyMem_RawMalloc(capacity ? 16*capacity : 1);
    if (!raw) { p->failed = 1; return; }
    int64_t *x = (int64_t *)raw;
    double *y = (double *)(raw + 8*capacity);
    p->error = decode(input, size, (char *)x, (char *)y, &count, &p->overflow);
    if (!p->error) {
        p->simple = x[0] >= INT64_MIN+4 && x[count-1] <= INT64_MAX-4;
        for (Py_ssize_t i = 1; p->simple && i < count; i++)
            p->simple = x[i] > x[i-1] && ((uint64_t)x[i] - (uint64_t)x[i-1]) % 4 == 0;
        /* Measured m/z is staged in the framed m/z storage (count <= framed points). */
        p->used = p->simple ? framed_size(x, count) : count;
        p->mz = PyMem_RawMalloc(8*p->used);
        p->y = p->simple ? PyMem_RawMalloc(8*p->used) : NULL;
        p->failed = !p->mz || (p->simple && !p->y);
    }
    if (!p->error && !p->failed) {
        measure(x, y, count, a, b, tdc, p->mz, &p->total, &p->top, &p->exact, &p->valid);
        p->first = p->mz[0];
        p->last = p->mz[count-1];
        p->complete = p->exact && p->valid && p->simple;
        if (p->complete) {
            p->used = frame(x, y, count, a, b, tdc, p->mz, p->y);
            if (denominator != 0.0) {
                if (!axis(p->mz, p->used)) p->status = 1;
                else {
                    for (Py_ssize_t i = 0; i < p->used; i++) p->mz[i] /= denominator;
                    if (!axis(p->mz, p->used)) p->status = 2;
                }
            }
        }
    }
    PyMem_RawFree(raw);
}

/* Results share a head for the reader's guards: (total, top, first m/z, last m/z,
 * valid, scale status, tail). An incomplete profile gives an exception, None (NumPy
 * decides) or a rejected head with a None tail; NULL without an exception means
 * complete. */
static PyObject *incomplete(const Profile *p)
{
    if (p->error) {
        PyErr_SetString(p->overflow ? PyExc_OverflowError : PyExc_ValueError, p->error);
        return NULL;
    }
    if (p->failed) return PyErr_NoMemory();
    if (!p->exact || (p->valid && !p->simple)) return Py_NewRef(Py_None);
    if (!p->valid) return Py_BuildValue("ddddOiO", p->total, p->top, p->first, p->last, Py_False, 0, Py_None);
    return NULL;
}

/* The complete head with tail (stolen; NULL propagates an exception). */
static PyObject *complete(const Profile *p, PyObject *tail)
{
    return tail ? Py_BuildValue("ddddOiN", p->total, p->top, p->first, p->last, Py_True, p->status, tail) : NULL;
}

static int parse_block(PyObject *args, PyObject **block, double *a, double *b, double *tdc, double *denominator)
{
    if (!PyArg_ParseTuple(args, "Sdddd", block, a, b, tdc, denominator)) return -1;
    if (PyBytes_GET_SIZE(*block) > PY_SSIZE_T_MAX/64) { PyErr_NoMemory(); return -1; }
    return 0;
}

/* reader.arrays(scan), then MassCorrection.correct for a nonzero denominator, with
 * one GIL release; tail (m/z, intensity) Buffers. */
static PyObject *profile_spectrum(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *block, *result;
    double a, b, tdc, denominator;
    if (parse_block(args, &block, &a, &b, &tdc, &denominator) < 0) return NULL;
    Profile p;
    Py_BEGIN_ALLOW_THREADS
    profile_block((const unsigned char *)PyBytes_AS_STRING(block), PyBytes_GET_SIZE(block), a, b, tdc, denominator, &p);
    Py_END_ALLOW_THREADS
    if ((result = incomplete(&p)) || PyErr_Occurred()) {
        PyMem_RawFree(p.mz); PyMem_RawFree(p.y);
        return result;
    }
    PyObject *mz = adopt((char *)p.mz, 8*p.used), *y = mz ? adopt((char *)p.y, 8*p.used) : (PyMem_RawFree(p.y), NULL);
    if (!y) { Py_XDECREF(mz); return NULL; }
    return complete(&p, Py_BuildValue("NN", mz, y));
}

/* kmzcore.sparse_profile.candidate_regions after validation, into one allocation
 * of starts, ends, apex, lower and upper (found values each): 1 where the
 * reference raises, -1 on memory failure. */
static int candidates_of(const double *mz, const double *y, Py_ssize_t n, double *unit, Py_ssize_t *found, char **parts)
{
    Py_ssize_t capacity = n/4 + 1;
    *found = 0;
    *parts = NULL;
    if (invalid(mz, y, n)) return 1;
    if (event_unit(y, n, unit)) return -1;
    char *scratch = PyMem_RawMalloc(40*capacity);
    if (!scratch) return -1;
    int64_t *starts = (int64_t *)scratch, *ends = starts + capacity;
    double *apex = (double *)(ends + capacity), *lower = apex + capacity, *upper = lower + capacity;
    *found = regions(mz, y, n, *unit, starts, ends, apex, lower, upper);
    if ((*parts = PyMem_RawMalloc(*found ? 40 * *found : 1)))
        for (int k = 0; k < 5; k++) memcpy(*parts + 8 * *found * k, scratch + 8*capacity*k, 8 * *found);
    PyMem_RawFree(scratch);
    return *parts ? 0 : -1;
}

/* candidate_regions(*reader_arrays(...)) with one GIL release; tail (event unit,
 * regions, Buffer of starts, ends, apex, lower, upper). None also where
 * candidate_regions raises. */
static PyObject *profile_regions(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *block, *result;
    double a, b, tdc, denominator, unit = 0.0;
    if (parse_block(args, &block, &a, &b, &tdc, &denominator) < 0) return NULL;
    Profile p;
    Py_ssize_t found = 0;
    char *parts = NULL;
    int state = 0;
    Py_BEGIN_ALLOW_THREADS
    profile_block((const unsigned char *)PyBytes_AS_STRING(block), PyBytes_GET_SIZE(block), a, b, tdc, denominator, &p);
    if (p.complete && !p.status) state = candidates_of(p.mz, p.y, p.used, &unit, &found, &parts);
    Py_END_ALLOW_THREADS
    PyMem_RawFree(p.mz); PyMem_RawFree(p.y);
    if ((result = incomplete(&p)) || PyErr_Occurred()) { PyMem_RawFree(parts); return result; }
    if (state < 0) return PyErr_NoMemory();
    if (state) return Py_NewRef(Py_None);
    if (p.status) return complete(&p, Py_NewRef(Py_None));
    PyObject *buffer = adopt(parts, 40*found);
    return buffer ? complete(&p, Py_BuildValue("dnN", unit, found, buffer)) : NULL;
}

/* kmzcore.sparse_profile.source_sampling_check (v3) for this spectrum's run-level
 * choice of candidate regions (0/1 bytes; gap_connections, candidate_regions and
 * region_indices), and the retained subset in kept (m/z, then intensity). */
typedef struct { double unit; Py_ssize_t selected, chosen, rescued, retained; double *kept; } Sample;

/* 0, 1 where the reference decides (it raises, or fewer than two points), -1 on memory failure. */
static int sample(const double *mz, const double *y, Py_ssize_t n, const char *pick, Py_ssize_t wanted, Sample *s)
{
    memset(s, 0, sizeof *s);
    if (n < 2 || n > PY_SSIZE_T_MAX/64) return 1;
    Py_ssize_t m = n-1, capacity = n/4 + 1, found;
    /* delta and window estimates, starts/ends/apex/lower/upper, connections and mask. */
    char *scratch = PyMem_RawMalloc(16*m + 40*capacity + m + n + 2);
    if (!scratch) return -1;
    double *delta = (double *)scratch, *full = delta + m;
    int64_t *starts = (int64_t *)(full + m), *ends = starts + capacity;
    double *apex = (double *)(ends + capacity), *lower = apex + capacity, *upper = lower + capacity;
    char *connected = (char *)(upper + capacity), *mask = connected + m;
    int state = invalid(mz, y, n) != NULL;
    if (!state) {
        gaps(mz, n, delta, full, connected);
        for (Py_ssize_t i = 0; !state && i < m; i++) state = !connected[i] && (y[i] > 0 || y[i+1] > 0);
    }
    if (!state) state = event_unit(y, n, &s->unit) ? -1 : 0;
    if (!state) {
        found = regions(mz, y, n, s->unit, starts, ends, apex, lower, upper);
        state = found != wanted;
    }
    if (!state) {
        double trigger = 20.0 * s->unit;
        memset(mask, 0, n+2);
        for (Py_ssize_t r = 0; r < found; r++) {
            if (!pick[r]) continue;
            s->chosen++;
            s->rescued += apex[r] <= trigger;
            s->selected += ends[r] - starts[r];
            memset(mask + starts[r] + 1, 1, ends[r] - starts[r]);
        }
        for (Py_ssize_t i = 0; i < n; i++) s->retained += mask[i] | mask[i+1] | mask[i+2];
        if (!(s->kept = PyMem_RawMalloc(s->retained ? 16*s->retained : 1))) state = -1;
        else
            for (Py_ssize_t i = 0, k = 0; i < n; i++)
                if (mask[i] | mask[i+1] | mask[i+2]) { s->kept[k] = mz[i]; s->kept[s->retained + k] = y[i]; k++; }
    }
    PyMem_RawFree(scratch);
    return state;
}

/* zlib.compress(data) as CPython's zlib_compress_impl configures the stream (level
 * Z_DEFAULT_COMPRESSION, MAX_WBITS, memory level 8, default strategy; all input
 * with Z_FINISH, so output chunking cannot change the bytes), then base64.
 * NULL when one deflateBound pass cannot complete. */
static char *deflate64(const void *in, Py_ssize_t n, Py_ssize_t *size)
{
    z_stream z;
    memset(&z, 0, sizeof z);
    if ((uint64_t)n > UINT_MAX/2 || deflateInit2(&z, Z_DEFAULT_COMPRESSION, Z_DEFLATED, MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return NULL;
    uLong bound = deflateBound(&z, (uLong)n);
    unsigned char *packed = bound <= UINT_MAX ? PyMem_RawMalloc(bound) : NULL;
    char *text = NULL;
    if (packed) {
        z.next_in = (Bytef *)in;
        z.avail_in = (uInt)n;
        z.next_out = packed;
        z.avail_out = (uInt)bound;
        if (deflate(&z, Z_FINISH) == Z_STREAM_END) {
            *size = (Py_ssize_t)(z.total_out + 2) / 3 * 4;
            if ((text = PyMem_RawMalloc(*size))) base64(packed, (Py_ssize_t)z.total_out, (unsigned char *)text);
        }
    }
    deflateEnd(&z);
    PyMem_RawFree(packed);
    return text;
}

/* A spectrum as the converter writes it, with one GIL release: profile, the Sparse
 * Profile v3 subset for a run-level choice (bytes, or None for all points), then
 * deflate64 of m/z and intensity. Tail (source points, written points, m/z text,
 * intensity text, then event unit, selected points, chosen and rescued regions or
 * Nones). None also where the sampling reference decides or zlib fails. */
static PyObject *encode_spectrum(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *block, *chosen, *result;
    double a, b, tdc, denominator;
    if (!PyArg_ParseTuple(args, "SddddO", &block, &a, &b, &tdc, &denominator, &chosen)) return NULL;
    if (PyBytes_GET_SIZE(block) > PY_SSIZE_T_MAX/64) return PyErr_NoMemory();
    if (chosen != Py_None && !PyBytes_Check(chosen)) {
        PyErr_SetString(PyExc_TypeError, "encode_spectrum requires bytes or None");
        return NULL;
    }
    const char *pick = chosen == Py_None ? NULL : PyBytes_AS_STRING(chosen);
    Py_ssize_t wanted = chosen == Py_None ? 0 : PyBytes_GET_SIZE(chosen), points = 0, sizes[2];
    Profile p;
    Sample s = {0};
    char *texts[2] = {NULL, NULL};
    int state = 0;
    Py_BEGIN_ALLOW_THREADS
    profile_block((const unsigned char *)PyBytes_AS_STRING(block), PyBytes_GET_SIZE(block), a, b, tdc, denominator, &p);
    if (p.complete && !p.status) {
        const double *mz = p.mz, *y = p.y;
        points = p.used;
        if (pick && !(state = sample(p.mz, p.y, p.used, pick, wanted, &s))) {
            mz = s.kept;
            y = s.kept + s.retained;
            points = s.retained;
        }
        if (!state && (!(texts[0] = deflate64(mz, 8*points, &sizes[0])) || !(texts[1] = deflate64(y, 8*points, &sizes[1]))))
            state = 1;
    }
    Py_END_ALLOW_THREADS
    PyMem_RawFree(p.mz); PyMem_RawFree(p.y); PyMem_RawFree(s.kept);
    if ((result = incomplete(&p)) || PyErr_Occurred() || state || p.status) {
        PyMem_RawFree(texts[0]); PyMem_RawFree(texts[1]);
        if (result || PyErr_Occurred()) return result;
        if (state < 0) return PyErr_NoMemory();
        return state ? Py_NewRef(Py_None) : complete(&p, Py_NewRef(Py_None));
    }
    PyObject *mz = adopt(texts[0], sizes[0]), *y = mz ? adopt(texts[1], sizes[1]) : (PyMem_RawFree(texts[1]), NULL);
    if (!y) { Py_XDECREF(mz); return NULL; }
    PyObject *tail = pick ? Py_BuildValue("nnNNdnnn", p.used, points, mz, y, s.unit, s.selected, s.chosen, s.rescued)
                          : Py_BuildValue("nnNNOOOO", p.used, points, mz, y, Py_None, Py_None, Py_None, Py_None);
    return complete(&p, tail);
}

/* base64.b64encode(zlib.compress(data)) of up to four immutable bytes with one GIL
 * release, as a tuple of Buffers; None if zlib fails. */
static PyObject *deflate_base64(PyObject *self, PyObject *args)
{
    (void)self;
    enum { MOST = 4 };
    Py_ssize_t count = PyTuple_GET_SIZE(args), lengths[MOST] = {0}, sizes[MOST] = {0};
    if (count > MOST) {
        PyErr_SetString(PyExc_TypeError, "deflate_base64 takes at most four buffers");
        return NULL;
    }
    const char *data[MOST];
    char *texts[MOST] = {NULL};
    PyObject *result = NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        if (!PyBytes_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "deflate_base64 requires immutable bytes");
            return NULL;
        }
        data[i] = PyBytes_AS_STRING(item);
        lengths[i] = PyBytes_GET_SIZE(item);
    }
    int failed = 0;
    /* Like zlib.compress, always without the GIL: deflate setup alone outweighs the hand-off. */
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < count; i++)
        failed |= !(texts[i] = deflate64(data[i], lengths[i], &sizes[i]));
    Py_END_ALLOW_THREADS
    if (failed) { result = Py_NewRef(Py_None); goto done; }
    if (!(result = PyTuple_New(count))) goto done;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *text = adopt(texts[i], sizes[i]);
        texts[i] = NULL;
        if (!text) { Py_CLEAR(result); goto done; }
        PyTuple_SET_ITEM(result, i, text);
    }
done:
    for (Py_ssize_t i = 0; i < count; i++) PyMem_RawFree(texts[i]);
    return result;
}

static PyObject *zlib_runtime(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    return PyUnicode_FromString(zlibVersion());
}

static PyMethodDef methods[] = {
    {"profile_spectrum", profile_spectrum, METH_VARARGS, "Guard head and framed (scaled) m/z and intensity Buffers of one immutable block, else None."},
    {"profile_regions", profile_regions, METH_VARARGS, "Guard head and Sparse Profile v3 candidate regions of one immutable block, else None."},
    {"encode_spectrum", encode_spectrum, METH_VARARGS, "Guard head and base64 zlib arrays (optionally the Sparse Profile v3 subset) of one block, else None."},
    {"deflate_base64", deflate_base64, METH_VARARGS, "base64 of zlib.compress for each immutable bytes argument, as Buffers, else None."},
    {"zlib_runtime", zlib_runtime, METH_NOARGS, "Runtime version of the zlib library linked here."},
    {"scan_metadata", scan_metadata, METH_VARARGS, "Calibration and checkpoints of a qualified scan preamble, else None."},
    {"decode_tokens", decode_tokens, METH_O, "Decode immutable WIFF bytes to owned native int64/float64 buffers."},
    {"rescue_selection", rescue_selection, METH_VARARGS, "Chosen flags (bytearray) for ordered finite candidate spans (bytes), else None."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__native(void)
{
    if (PyType_Ready(&BufferType) < 0) return NULL;
    return PyModule_Create(&module);
}
