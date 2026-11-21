/* Agilent MassHunter profile block decoders; equal to kmzconvert_core.agilent's
 * lzf_decompress and rle_decode references, which tests compare against. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* Raw LZF stream to exactly `size` bytes. Returns NULL with ValueError set. */
static PyObject *lzf_decompress(PyObject *self, PyObject *args)
{
    Py_buffer in;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "y*n", &in, &size)) return NULL;
    if (size < 0) { PyBuffer_Release(&in); PyErr_SetString(PyExc_ValueError, "Invalid LZF size"); return NULL; }
    PyObject *result = PyBytes_FromStringAndSize(NULL, size);
    if (!result) { PyBuffer_Release(&in); return NULL; }
    const uint8_t *ip = in.buf, *end = ip + in.len;
    uint8_t *out = (uint8_t *)PyBytes_AS_STRING(result), *op = out, *oend = out + size;
    const char *error = NULL;
    Py_BEGIN_ALLOW_THREADS
    while (ip < end) {
        unsigned control = *ip++;
        if (control < 32) {
            size_t run = control + 1;
            if ((size_t)(end - ip) < run) { error = "Truncated LZF literal"; break; }
            if ((size_t)(oend - op) < run) { error = "LZF output exceeds the declared size"; break; }
            memcpy(op, ip, run); op += run; ip += run;
        } else {
            size_t length = control >> 5;
            if (length == 7) {
                if (ip >= end) { error = "Truncated LZF back-reference"; break; }
                length += *ip++;
            }
            if (ip >= end) { error = "Truncated LZF back-reference"; break; }
            size_t back = ((size_t)(control & 0x1f) << 8) + *ip++ + 1;
            if (back > (size_t)(op - out)) { error = "Invalid LZF back-reference"; break; }
            length += 2;
            if ((size_t)(oend - op) < length) { error = "LZF output exceeds the declared size"; break; }
            const uint8_t *ref = op - back;
            for (size_t i = 0; i < length; i++) op[i] = ref[i]; /* overlapping copies repeat bytes */
            op += length;
        }
    }
    if (!error && op != oend) error = "LZF output is shorter than the declared size";
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&in);
    if (error) { Py_DECREF(result); PyErr_SetString(PyExc_ValueError, error); return NULL; }
    return result;
}

/* Run-length profile stream to `count` little-endian int64 intensities. */
static PyObject *rle_decode(PyObject *self, PyObject *args)
{
    Py_buffer in;
    Py_ssize_t start, count;
    if (!PyArg_ParseTuple(args, "y*nn", &in, &start, &count)) return NULL;
    if (count < 0 || start < 0 || start > count || count > PY_SSIZE_T_MAX / 8) {
        PyBuffer_Release(&in); PyErr_SetString(PyExc_ValueError, "Invalid run-length start point"); return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize(NULL, count * 8);
    if (!result) { PyBuffer_Release(&in); return NULL; }
    int64_t *out = (int64_t *)PyBytes_AS_STRING(result);
    const uint8_t *ip = in.buf, *end = ip + in.len;
    const char *error = NULL;
    Py_BEGIN_ALLOW_THREADS
    memset(out, 0, (size_t)count * 8);
    Py_ssize_t point = start;
    int width = 4;
    while (ip < end) {
        if (end - ip < width) { error = "Truncated run-length stream"; break; }
        int64_t value;
        switch (width) {
            case 1: { int8_t v; memcpy(&v, ip, 1); value = v; break; }
            case 2: { int16_t v; memcpy(&v, ip, 2); value = v; break; }
            case 4: { int32_t v; memcpy(&v, ip, 4); value = v; break; }
            default: { int64_t v; memcpy(&v, ip, 8); value = v; break; }
        }
        ip += width;
        if (value >= 0) {
            if (point >= count) { error = "Run-length stream exceeds the point count"; break; }
            out[point++] = value;
        } else {
            uint64_t u = (uint64_t)(-(value + 1));
            uint64_t skip = u >> 2;
            if (skip > (uint64_t)(count - point)) { error = "Run-length stream exceeds the point count"; break; }
            point += (Py_ssize_t)skip;
            width = 1 << (u & 3);
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&in);
    if (error) { Py_DECREF(result); PyErr_SetString(PyExc_ValueError, error); return NULL; }
    return result;
}

static PyMethodDef methods[] = {
    {"lzf_decompress", lzf_decompress, METH_VARARGS, "Raw LZF bytes to exactly the declared number of bytes."},
    {"rle_decode", rle_decode, METH_VARARGS, "Run-length profile stream to little-endian int64 intensities."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_agilent", NULL, -1, methods};
PyMODINIT_FUNC PyInit__agilent(void) { return PyModule_Create(&module); }
