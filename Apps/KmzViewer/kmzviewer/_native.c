#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <zlib.h>

/* Optional viewer helpers. Decoding releases the GIL and accepts only canonical
 * strict base64 and complete zlib streams of the expected size with no trailing
 * data; it returns None for anything else, so mzml_direct's binascii/zlib path
 * decides. The scanners and checks reproduce the Python expressions they replace. */

static unsigned char values[256];

/* Decode canonical base64 (alphabet only, one final padded quantum). Returns the
 * output length, or -1 if the text lies outside that subset. */
static Py_ssize_t decode64(const unsigned char *in, Py_ssize_t n, unsigned char *out)
{
    if (n <= 0 || n % 4)
        return -1;
    Py_ssize_t pad = in[n-1] == '=' ? (in[n-2] == '=' ? 2 : 1) : 0;
    Py_ssize_t full = n/4 - (pad != 0);
    unsigned bad = 0;
    for (Py_ssize_t q = 0; q < full; q++, in += 4, out += 3) {
        unsigned a = values[in[0]], b = values[in[1]], c = values[in[2]], d = values[in[3]];
        bad |= a | b | c | d;
        unsigned v = (a << 18) | (b << 12) | (c << 6) | d;
        out[0] = (unsigned char)(v >> 16);
        out[1] = (unsigned char)(v >> 8);
        out[2] = (unsigned char)v;
    }
    if (bad & 0x40)
        return -1;
    if (pad) {
        unsigned a = values[in[0]], b = values[in[1]], c = pad == 2 ? 0 : values[in[2]];
        /* Nonzero discarded bits are not canonical; leave them to binascii. */
        if ((a | b | c) & 0x40 || (pad == 2 ? b & 0x0f : c & 0x03))
            return -1;
        out[0] = (unsigned char)((a << 2) | (b >> 4));
        if (pad == 1)
            out[1] = (unsigned char)((b << 4) | (c >> 2));
    }
    return full*3 + (3 - pad) % 3;
}

/* Inflate one complete zlib stream into exactly `size` or fewer bytes. Returns the
 * output length, or -1 for any error, excess output or unused input. */
static Py_ssize_t inflate_all(unsigned char *in, Py_ssize_t n, unsigned char *out, Py_ssize_t size)
{
    z_stream stream = {0};
    if (n > UINT_MAX || size > UINT_MAX || inflateInit2(&stream, MAX_WBITS) != Z_OK)
        return -1;
    stream.next_in = in;
    stream.avail_in = (uInt)n;
    stream.next_out = out;
    stream.avail_out = (uInt)size;
    int status = inflate(&stream, Z_FINISH);
    Py_ssize_t done = status == Z_STREAM_END && stream.avail_in == 0 ? (Py_ssize_t)stream.total_out : -1;
    inflateEnd(&stream);
    return done;
}

/* inflate_base64(data, start, limit, compressed, size) -> (bytearray | None, end).
 * `end` is data.find(b"<", start). The text before it is decoded when it holds
 * 1..limit bytes; zlib output must fill at most `size` bytes. */
static PyObject *inflate_base64(PyObject *module, PyObject *args)
{
    Py_buffer view;
    Py_ssize_t start, limit, size, end = -1, length = -1;
    int compressed;
    if (!PyArg_ParseTuple(args, "y*nnpn", &view, &start, &limit, &compressed, &size))
        return NULL;
    /* The GIL is released while reading the input: accept only immutable bytes. */
    if (view.obj == NULL || !PyBytes_CheckExact(view.obj)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "inflate_base64 requires bytes");
        return NULL;
    }
    if (start < 0 || start > view.len || (compressed && size <= 0)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "invalid text start or output size");
        return NULL;
    }
    const unsigned char *buf = view.buf, *in = buf + start;
    PyObject *result = compressed ? PyByteArray_FromStringAndSize(NULL, size) : NULL;
    if (compressed && result == NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }
    unsigned char *out = result ? (unsigned char *)PyByteArray_AS_STRING(result) : NULL;
    /* data is an immutable bytes view and the result is not yet shared. */
    Py_BEGIN_ALLOW_THREADS
    const unsigned char *found = memchr(in, '<', view.len - start);
    end = found ? found - buf : -1;
    Py_ssize_t n = end - start;
    if (compressed && end > start && n <= limit) {
        unsigned char *scratch = PyMem_RawMalloc(n / 4 * 3 + 1);
        if (scratch != NULL) {
            length = decode64(in, n, scratch);
            if (length >= 0)
                length = inflate_all(scratch, length, out, size);
            PyMem_RawFree(scratch);
        }
    }
    Py_END_ALLOW_THREADS
    if (!compressed && end > start && end - start <= limit && (end - start) % 4 == 0) {
        Py_ssize_t n = end - start;
        result = PyByteArray_FromStringAndSize(NULL, n / 4 * 3);
        if (result == NULL) {
            PyBuffer_Release(&view);
            return NULL;
        }
        out = (unsigned char *)PyByteArray_AS_STRING(result);
        Py_BEGIN_ALLOW_THREADS
        length = decode64(in, n, out);
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&view);
    if (result != NULL && (length < 0 || (length != PyByteArray_GET_SIZE(result)
                                          && PyByteArray_Resize(result, length) < 0))) {
        Py_CLEAR(result);
        if (length >= 0)
            return NULL;  /* Resize failed. */
    }
    if (result == NULL)
        result = Py_NewRef(Py_None);
    return Py_BuildValue("Nn", result, end);
}

/* Read float64 values without assuming buffer alignment. */
static double item(const char *p, Py_ssize_t i)
{
    double value;
    memcpy(&value, p + 8 * i, 8);
    return value;
}

/* ordered(float64 buffer) -> 2 if every value is finite and nondecreasing, 1 if
 * finite only, 0 otherwise. Branch-free loops vectorize; they keep the GIL. */
static PyObject *ordered(PyObject *module, PyObject *arg)
{
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    const char *p = view.buf;
    Py_ssize_t n = view.len / 8;
    int finite = 1, sorted = 1;
    for (Py_ssize_t i = 0; i < n; i++)
        finite &= isfinite(item(p, i)) != 0;
    for (Py_ssize_t i = 1; i < n; i++)
        sorted &= item(p, i) >= item(p, i - 1);
    PyBuffer_Release(&view);
    return PyLong_FromLong(finite ? 1 + sorted : 0);
}

/* finite_max(float64 buffer) -> (finite count, largest finite value or 0.0).
 * Independent lanes vectorize; a nonzero maximum is one exact value in any order. */
static PyObject *finite_max(PyObject *module, PyObject *arg)
{
    enum { LANES = 8 };
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    const char *p = view.buf;
    Py_ssize_t n = view.len / 8, count = 0, counts[LANES] = {0}, i = 0;
    double best = -INFINITY, lanes[LANES];
    for (int k = 0; k < LANES; k++)
        lanes[k] = -INFINITY;
    for (; i + LANES <= n; i += LANES) {
        for (int k = 0; k < LANES; k++) {
            double value = item(p, i + k);
            int keep = isfinite(value) != 0;
            double candidate = keep ? value : -INFINITY;
            counts[k] += keep;
            lanes[k] = candidate > lanes[k] ? candidate : lanes[k];
        }
    }
    for (; i < n; i++) {
        double value = item(p, i);
        if (isfinite(value)) {
            count++;
            best = value > best ? value : best;
        }
    }
    for (int k = 0; k < LANES; k++) {
        count += counts[k];
        best = lanes[k] > best ? lanes[k] : best;
    }
    PyBuffer_Release(&view);
    return Py_BuildValue("nd", count, count ? best : 0.0);
}

/* \w and \s of a bytes pattern without flags. */
static int word(unsigned char c)
{
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static int space(unsigned char c)
{
    return c == ' ' || (c >= '\t' && c <= '\r');
}

/* (spectrum|chromatogram)\s within [p, limit), as in a bytes pattern. */
static int named(const unsigned char *p, const unsigned char *limit)
{
    static const char *const names[] = {"spectrum", "chromatogram"};
    for (int i = 0; i < 2; i++) {
        size_t n = strlen(names[i]);
        if (limit - p > (Py_ssize_t)n && !memcmp(p, names[i], n) && space(p[n]))
            return 1;
    }
    return 0;
}

/* ([(offset, next '<'), ...], last '<' or -1) for each '<'-delimited chunk before
 * the final '<' that matches <(?:\w+:)?(spectrum|chromatogram)\s; mzml_direct
 * re-checks each with that regex. */
static PyObject *indexed_tags(PyObject *module, PyObject *args)
{
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view))
        return NULL;
    const unsigned char *buf = view.buf, *stop = buf + view.len;
    const unsigned char *at = memchr(buf, '<', view.len), *last = at;
    PyObject *found = PyList_New(0);
    while (found != NULL && at != NULL) {
        const unsigned char *following = memchr(at + 1, '<', stop - at - 1);
        if (following == NULL)
            break;
        const unsigned char *name = at + 1, *w = name;
        while (w < following && word(*w))
            w++;
        if ((w > name && *w == ':' && named(w + 1, following)) || named(name, following)) {
            PyObject *pair = Py_BuildValue("nn", (Py_ssize_t)(at - buf), (Py_ssize_t)(following - buf));
            if (pair == NULL || PyList_Append(found, pair) < 0)
                Py_CLEAR(found);
            Py_XDECREF(pair);
        }
        last = at = following;
    }
    PyBuffer_Release(&view);
    return found == NULL ? NULL : Py_BuildValue("Nn", found, last ? (Py_ssize_t)(last - buf) : (Py_ssize_t)-1);
}

/* Exactly buffer.find(needle, start), stepping between first-byte occurrences. */
static PyObject *find(PyObject *module, PyObject *args)
{
    Py_buffer view, needle;
    Py_ssize_t start = 0, result = -1;
    if (!PyArg_ParseTuple(args, "y*y*|n", &view, &needle, &start))
        return NULL;
    const unsigned char *buf = view.buf, *key = needle.buf;
    Py_ssize_t n = needle.len;
    if (start < 0)
        start = start + view.len < 0 ? 0 : start + view.len;
    if (n == 0) {
        result = start <= view.len ? start : -1;
    } else if (n <= view.len - start) {
        const unsigned char *at = buf + start, *last = buf + view.len - n;
        while (at <= last && (at = memchr(at, key[0], last - at + 1)) != NULL) {
            if (!memcmp(at, key, n)) {
                result = at - buf;
                break;
            }
            at++;
        }
    }
    PyBuffer_Release(&view);
    PyBuffer_Release(&needle);
    return PyLong_FromSsize_t(result);
}

/* format(v, ".10g") into out (at least 32 bytes); returns the length. Finite
 * values printed in fixed notation use exact 128-bit integer scaling with the
 * same round-half-even as CPython's dtoa; all others call PyOS_double_to_string,
 * which is what format() calls. */
static Py_ssize_t format10(double v, char *out)
{
#ifdef __SIZEOF_INT128__
    static const uint64_t pow10[] = {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL,
        10000000ULL, 100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL,
        10000000000000ULL, 100000000000000ULL, 1000000000000000ULL};
    double a = fabs(v);
    if (isfinite(a) && a >= 1e-5 && a < 1e10) {
        uint64_t bits;
        memcpy(&bits, &a, 8);
        uint64_t m = (bits & ((1ULL << 52) - 1)) | (1ULL << 52);
        int s = 1075 - (int)(bits >> 52);  /* a = m / 2**s, 19 <= s <= 69 here */
        int d = (int)floor(log10(a));
        for (int attempt = 0; attempt < 4 && 9 - d >= 0 && 9 - d <= 15; attempt++) {
            unsigned __int128 n = (unsigned __int128)m * pow10[9 - d], q = n >> s;
            if (q >= 10000000000ULL || q < 1000000000ULL) {
                d += q >= 10000000000ULL ? 1 : -1;
                continue;
            }
            unsigned __int128 r = n - (q << s), half = (unsigned __int128)1 << (s - 1);
            if (r > half || (r == half && (q & 1)))
                q++;
            if (q == 10000000000ULL) {
                q = 1000000000ULL;
                d++;
            }
            if (d < -4 || d > 9)
                break;
            char digits[10];
            uint64_t value = (uint64_t)q;
            for (int i = 9; i >= 0; i--, value /= 10)
                digits[i] = (char)('0' + value % 10);
            int length = 10, point = d + 1;
            while (digits[length - 1] == '0')
                length--;
            char *p = out;
            if (v < 0)
                *p++ = '-';
            if (point <= 0) {
                memcpy(p, "0.0000", 2 - point);
                p += 2 - point;
                memcpy(p, digits, length);
                p += length;
            } else if (point >= length) {
                memcpy(p, digits, length);
                memset(p + length, '0', point - length);
                p += point;
            } else {
                memcpy(p, digits, point);
                p[point] = '.';
                memcpy(p + point + 1, digits + point, length - point);
                p += length + 1;
            }
            return p - out;
        }
    }
#endif
    char *text = PyOS_double_to_string(v, 'g', 10, 0, NULL);
    if (text == NULL)
        return -1;
    Py_ssize_t length = (Py_ssize_t)strlen(text);
    memcpy(out, text, length);
    PyMem_Free(text);
    return length;
}

/* path_data(float64 buffer of x, y pairs) -> "M x y L x y ..." with every value
 * formatted exactly as format(value, ".10g"). */
static PyObject *path_data(PyObject *module, PyObject *arg)
{
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    const char *p = view.buf;
    Py_ssize_t n = view.len / 8;
    if (view.len % 16) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "path_data needs whole float64 x, y pairs");
        return NULL;
    }
    char *text = PyMem_Malloc(n * 36 + 2), *out = text;
    if (text == NULL) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    *out++ = 'M';
    *out++ = ' ';
    for (Py_ssize_t i = 0; i < n; i++) {
        if (i % 2)
            *out++ = ' ';
        else if (i) {
            memcpy(out, " L ", 3);
            out += 3;
        }
        Py_ssize_t length = format10(item(p, i), out);
        if (length < 0) {
            PyMem_Free(text);
            PyBuffer_Release(&view);
            return NULL;
        }
        out += length;
    }
    PyBuffer_Release(&view);
    PyObject *result = PyUnicode_DecodeASCII(text, out - text, NULL);
    PyMem_Free(text);
    return result;
}

static PyMethodDef methods[] = {
    {"inflate_base64", inflate_base64, METH_VARARGS,
     "inflate_base64(data, start, limit, compressed, size) -> (bytearray | None, end)"},
    {"ordered", ordered, METH_O, "ordered(float64 buffer) -> 0 | 1 | 2"},
    {"finite_max", finite_max, METH_O, "finite_max(float64 buffer) -> (count, maximum)"},
    {"indexed_tags", indexed_tags, METH_VARARGS, "indexed_tags(buffer) -> (list[(int, int)], int)"},
    {"find", find, METH_VARARGS, "find(buffer, needle, start=0) -> int"},
    {"path_data", path_data, METH_O, "path_data(float64 x, y buffer) -> 'M x y L x y ...'"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods};

PyMODINIT_FUNC PyInit__native(void)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    memset(values, 0x40, sizeof values);
    for (unsigned i = 0; i < 64; i++)
        values[(unsigned char)alphabet[i]] = (unsigned char)i;
    return PyModule_Create(&module);
}
