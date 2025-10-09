package com.github.lambdv.primitives;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import java.util.List;

public class Tensor {
    private final double[] data;
    private final int[] shape;
    private final int[] strides;
    private final int size;

    // Constructors
    public Tensor(int... shape) {
        validateShape(shape);
        this.shape = shape.clone();
        this.size = product(shape);
        this.data = new double[this.size];
        this.strides = defaultStrides(this.shape);
    }

    public Tensor(double[] data, int... shape) {
        validateShape(shape);
        int expected = product(shape);
        if (data.length != expected) {
            throw new IllegalArgumentException("Data length " + data.length + " does not match shape product " + expected);
        }
        this.shape = shape.clone();
        this.size = expected;
        this.data = data.clone();
        this.strides = defaultStrides(this.shape);
    }

    private Tensor(double[] data, int[] shape, int[] strides) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        this.size = product(shape);
    }

    // Factories
    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public static Tensor full(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, value);
        return t;
    }

    public static Tensor of(double[] data, int... shape) {
        return new Tensor(data, shape);
    }

    // Random / numeric factories
    public static Tensor rand(int... shape) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        for (int i = 0; i < t.data.length; i++) t.data[i] = r.nextDouble();
        return t;
    }

    public static Tensor randn(int... shape) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        for (int i = 0; i < t.data.length; i++) t.data[i] = r.nextGaussian();
        return t;
    }

    public static Tensor eye(int n) {
        Tensor t = new Tensor(n, n);
        for (int i = 0; i < n; i++) t.data[i * n + i] = 1.0;
        return t;
    }

    public static Tensor arange(int end) {
        if (end < 0) throw new IllegalArgumentException("end must be >= 0");
        double[] data = new double[end];
        for (int i = 0; i < end; i++) data[i] = i;
        return new Tensor(data, end);
    }

    public static Tensor arange(double start, double end, double step) {
        if (step == 0.0) throw new IllegalArgumentException("step must be non-zero");
        int len = (int) Math.max(0, Math.ceil((end - start) / step));
        double[] data = new double[len];
        double v = start;
        for (int i = 0; i < len; i++, v += step) data[i] = v;
        return new Tensor(data, len);
    }

    public static Tensor linspace(double start, double end, int steps) {
        return linspace(start, end, steps, true);
    }

    public static Tensor linspace(double start, double end, int steps, boolean endpoint) {
        if (steps <= 0) throw new IllegalArgumentException("steps must be > 0");
        double[] data = new double[steps];
        if (steps == 1) {
            data[0] = start;
        } else {
            double denom = endpoint ? (steps - 1) : steps;
            for (int i = 0; i < steps; i++) data[i] = start + (end - start) * (i / denom);
        }
        return new Tensor(data, steps);
    }

    // Basic info
    public int ndim() { return shape.length; }
    public int size() { return size; }
    public int[] shape() { return shape.clone(); }
    public int[] strides() { return strides.clone(); }
    public double[] toArray() { return toContiguousArray(); }

    // Indexing
    public double get(int... indices) {
        int off = offset(indices);
        return data[off];
    }

    public Tensor getSlice(int... indices) {
        if (indices == null) throw new IllegalArgumentException("indices must not be null");
        if (indices.length < 0 || indices.length >= shape.length) {
            throw new IllegalArgumentException("Slice requires fewer indices than rank; got " + indices.length + " for rank " + shape.length);
        }
        for (int d = 0; d < indices.length; d++) {
            int idx = indices[d];
            int dimSize = shape[d];
            if (idx < 0 || idx >= dimSize) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for dim " + d + " with size " + dimSize);
            }
        }

        int outRank = shape.length - indices.length;
        int[] outShape = new int[outRank];
        for (int i = 0; i < outRank; i++) outShape[i] = shape[i + indices.length];

        double[] out = new double[product(outShape)];
        int write = 0;
        IndexIterator it = new IndexIterator(outShape);
        while (it.hasNext()) {
            int[] tail = it.next();
            int[] full = new int[shape.length];
            // prefix fixed indices
            for (int i = 0; i < indices.length; i++) full[i] = indices[i];
            // tail indices
            for (int i = 0; i < tail.length; i++) full[i + indices.length] = tail[i];
            out[write++] = get(full);
        }
        return new Tensor(out, outShape);
    }

    public void set(double value, int... indices) {
        int off = offset(indices);
        data[off] = value;
    }

    private int offset(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Index rank " + indices.length + " must match tensor rank " + shape.length);
        }
        int off = 0;
        for (int d = 0; d < shape.length; d++) {
            int idx = indices[d];
            if (idx < 0 || idx >= shape[d]) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for dim " + d + " with size " + shape[d]);
            }
            off += idx * strides[d];
        }
        return off;
    }

    // Reshape / Permute
    public Tensor reshape(int... newShape) {
        validateShape(newShape);
        int inferred = -1;
        long knownProduct = 1;
        for (int i = 0; i < newShape.length; i++) {
            int d = newShape[i];
            if (d == -1) {
                if (inferred != -1) throw new IllegalArgumentException("Only one dimension can be inferred with -1");
                inferred = i;
            } else {
                if (d < 0) throw new IllegalArgumentException("Shape values must be >= -1");
                knownProduct *= d;
            }
        }
        int[] finalShape = newShape.clone();
        if (inferred != -1) {
            if (this.size % knownProduct != 0) throw new IllegalArgumentException("Cannot infer dimension: size not divisible");
            finalShape[inferred] = (int) (this.size / knownProduct);
        }
        if (product(finalShape) != this.size) throw new IllegalArgumentException("Cannot reshape: total size mismatch");
        return new Tensor(this.data, finalShape, defaultStrides(finalShape));
    }

    public Tensor view(int... newShape) { return reshape(newShape); }

    public Tensor unsqueeze(int dim) {
        if (dim < 0 || dim > shape.length) throw new IllegalArgumentException("Invalid dim for unsqueeze");
        int[] newShape = new int[shape.length + 1];
        int[] newStrides = new int[strides.length + 1];
        for (int i = 0, j = 0; i < newShape.length; i++) {
            if (i == dim) {
                newShape[i] = 1;
                newStrides[i] = (dim < strides.length) ? strides[dim] : 1;
            } else {
                newShape[i] = shape[j];
                newStrides[i] = strides[j];
                j++;
            }
        }
        return new Tensor(this.data, newShape, newStrides);
    }

    public Tensor squeeze() { return squeezeAll(); }

    public Tensor squeeze(int dim) {
        if (dim < 0 || dim >= shape.length) throw new IllegalArgumentException("Invalid dim for squeeze");
        if (shape[dim] != 1) return this;
        int[] newShape = new int[shape.length - 1];
        int[] newStrides = new int[strides.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i == dim) continue;
            newShape[j] = shape[i];
            newStrides[j] = strides[i];
            j++;
        }
        return new Tensor(this.data, newShape, newStrides);
    }

    private Tensor squeezeAll() {
        int keep = 0;
        for (int s : shape) if (s != 1) keep++;
        if (keep == shape.length) return this;
        if (keep == 0) return this; // keep at least 1D for simplicity
        int[] newShape = new int[keep];
        int[] newStrides = new int[keep];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (shape[i] == 1) continue;
            newShape[j] = shape[i];
            newStrides[j] = strides[i];
            j++;
        }
        return new Tensor(this.data, newShape, newStrides);
    }

    public Tensor flatten() { return reshape(size); }

    public Tensor contiguous() { return new Tensor(toContiguousArray(), shape.clone()); }

    public Tensor permute(int... dims) {
        if (dims.length != shape.length) {
            throw new IllegalArgumentException("Permutation rank must match tensor rank");
        }
        boolean[] seen = new boolean[dims.length];
        int[] newShape = new int[dims.length];
        int[] newStrides = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            int d = dims[i];
            if (d < 0 || d >= shape.length || seen[d]) {
                throw new IllegalArgumentException("Invalid permutation");
            }
            seen[d] = true;
            newShape[i] = shape[d];
            newStrides[i] = strides[d];
        }
        return new Tensor(this.data, newShape, newStrides);
    }

    public Tensor T() {
        if (ndim() != 2) {
            throw new IllegalStateException("T() is only valid for 2D tensors");
        }
        return permute(1, 0);
    }

    // Unary ops
    public Tensor map(DoubleUnaryOperator op) {
        double[] out = new double[size];
        int idx = 0;
        IndexIterator it = new IndexIterator(shape);
        while (it.hasNext()) {
            int[] ix = it.next();
            out[idx++] = op.applyAsDouble(get(ix));
        }
        return new Tensor(out, shape.clone());
    }

    public Tensor add(double scalar) { return map(a -> a + scalar); }
    public Tensor sub(double scalar) { return map(a -> a - scalar); }
    public Tensor mul(double scalar) { return map(a -> a * scalar); }
    public Tensor div(double scalar) { return map(a -> a / scalar); }

    public Tensor exp() { return map(Math::exp); }
    public Tensor log() { return map(Math::log); }
    public Tensor tanh() { return map(Math::tanh); }
    public Tensor relu() { return map(a -> a > 0 ? a : 0.0); }
    public Tensor sigmoid() { return map(a -> 1.0 / (1.0 + Math.exp(-a))); }

    // Elementwise binary ops with broadcasting
    public Tensor add(Tensor other) { return elementwise(other, (a, b) -> a + b); }
    public Tensor sub(Tensor other) { return elementwise(other, (a, b) -> a - b); }
    public Tensor mul(Tensor other) { return elementwise(other, (a, b) -> a * b); }
    public Tensor div(Tensor other) { return elementwise(other, (a, b) -> a / b); }

    private Tensor elementwise(Tensor other, DoubleBinaryOperator op) {
        int[] outShape = broadcastShape(this.shape, other.shape);
        double[] out = new double[product(outShape)];

        int[] aStridesForOut = broadcastStridesFor(this, outShape);
        int[] bStridesForOut = broadcastStridesFor(other, outShape);

        int idx = 0;
        IndexIterator it = new IndexIterator(outShape);
        while (it.hasNext()) {
            int[] ix = it.next();
            int offA = dot(ix, aStridesForOut);
            int offB = dot(ix, bStridesForOut);
            out[idx++] = op.applyAsDouble(this.data[offA], other.data[offB]);
        }
        return new Tensor(out, outShape);
    }

    // Matrix multiplication (2D x 2D)
    public Tensor matmul(Tensor other) {
        if (this.ndim() != 2 || other.ndim() != 2) {
            throw new IllegalArgumentException("matmul requires two 2D tensors");
        }
        int m = this.shape[0];
        int kA = this.shape[1];
        int kB = other.shape[0];
        int n = other.shape[1];
        if (kA != kB) {
            throw new IllegalArgumentException("Inner dimensions must match for matmul");
        }
        double[] out = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < kA; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                out[i * n + j] = sum;
            }
        }
        return new Tensor(out, m, n);
    }

    public Tensor mm(Tensor other) { return matmul(other); }

    public double dot(Tensor other) {
        if (this.ndim() != 1 || other.ndim() != 1)
            throw new IllegalArgumentException("dot requires two 1D tensors");
        if (this.size != other.size)
            throw new IllegalArgumentException("Vectors must be the same length");
        double s = 0.0;
        for (int i = 0; i < size; i++) s += this.get(i) * other.get(i);
        return s;
    }

    // Reductions
    public double sum() {
        double s = 0.0;
        IndexIterator it = new IndexIterator(shape);
        while (it.hasNext()) {
            s += get(it.next());
        }
        return s;
    }

    public double mean() {
        return size == 0 ? Double.NaN : sum() / size;
        }

    public Tensor sum(int dim) { return sum(dim, false); }

    public Tensor sum(int dim, boolean keepdim) {
        return reduceAlongDim(dim, keepdim, Double::sum, 0.0);
    }

    public Tensor mean(int dim) { return mean(dim, false); }

    public Tensor mean(int dim, boolean keepdim) {
        Tensor s = sum(dim, keepdim);
        double denom = this.shape[dim];
        return s.div(denom);
    }

    private Tensor reduceAlongDim(int dim, boolean keepdim, DoubleBinaryOperator reducer, double identity) {
        if (dim < 0 || dim >= shape.length) throw new IllegalArgumentException("Invalid reduction dim");
        int[] outShape = keepdim ? shape.clone() : new int[shape.length - 1];
        if (keepdim) outShape[dim] = 1; else {
            for (int i = 0, j = 0; i < shape.length; i++) if (i != dim) outShape[j++] = shape[i];
        }
        int outSize = product(outShape);
        double[] out = new double[outSize];
        Arrays.fill(out, identity);
        int[] outStrides = defaultStrides(outShape);

        IndexIterator it = new IndexIterator(shape);
        while (it.hasNext()) {
            int[] ix = it.next();
            int[] outIx;
            if (keepdim) {
                outIx = ix.clone();
                outIx[dim] = 0;
            } else {
                outIx = new int[outShape.length];
                for (int i = 0, j = 0; i < ix.length; i++) if (i != dim) outIx[j++] = ix[i];
            }
            int offOut = dot(outIx, outStrides);
            double cur = out[offOut];
            double val = get(ix);
            out[offOut] = reducer.applyAsDouble(cur, val);
        }
        return new Tensor(out, outShape);
    }

    // Utilities
    private static void validateShape(int[] shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape must be non-empty");
        }
        for (int s : shape) {
            if (s < 0) throw new IllegalArgumentException("Shape dimensions must be >= 0");
        }
    }

    private static int product(int[] dims) {
        int p = 1;
        for (int d : dims) p *= d;
        return p;
    }

    private static int[] defaultStrides(int[] shape) {
        int r = shape.length;
        int[] s = new int[r];
        int stride = 1;
        for (int i = r - 1; i >= 0; i--) {
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }

    private static int[] broadcastShape(int[] a, int[] b) {
        int ra = a.length;
        int rb = b.length;
        int r = Math.max(ra, rb);
        int[] out = new int[r];
        for (int i = 0; i < r; i++) {
            int ad = (i < r - ra) ? 1 : a[i - (r - ra)];
            int bd = (i < r - rb) ? 1 : b[i - (r - rb)];
            if (ad != bd && ad != 1 && bd != 1) {
                throw new IllegalArgumentException("Shapes are not broadcastable: " + Arrays.toString(a) + " vs " + Arrays.toString(b));
            }
            out[i] = Math.max(ad, bd);
        }
        return out;
    }

    private static int[] broadcastStridesFor(Tensor t, int[] outShape) {
        int rt = t.shape.length;
        int ro = outShape.length;
        int[] outStrides = new int[ro];
        for (int i = 0; i < ro; i++) {
            int td = (i < ro - rt) ? 1 : t.shape[i - (ro - rt)];
            int stride = (i < ro - rt) ? 0 : t.strides[i - (ro - rt)];
            outStrides[i] = (td == 1 && outShape[i] > 1) ? 0 : stride;
        }
        return outStrides;
    }

    private static int dot(int[] a, int[] b) {
        int s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private double[] toContiguousArray() {
        if (isContiguous()) {
            return data.clone();
        }
        double[] out = new double[size];
        int idx = 0;
        IndexIterator it = new IndexIterator(shape);
        while (it.hasNext()) {
            out[idx++] = get(it.next());
        }
        return out;
    }

    private boolean isContiguous() {
        return Arrays.equals(strides, defaultStrides(shape));
    }

    public double item() {
        if (size != 1) throw new IllegalStateException("item() only valid for a single-element tensor");
        return toContiguousArray()[0];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape)).append(", data=");
        int limit = Math.min(size, 32);
        sb.append("[");
        if (limit > 0) {
            int shown = 0;
            IndexIterator it = new IndexIterator(shape);
            while (it.hasNext() && shown < limit) {
                if (shown > 0) sb.append(", ");
                sb.append(get(it.next()));
                shown++;
            }
            if (limit < size) sb.append(", ...");
        }
        sb.append("])");
        return sb.toString();
    }
    /**
     * returns string representation of the data shaped as the shape of the tensor
     * @return
     */
    public String toStringShape(){
        StringBuilder sb = new StringBuilder();
        appendToStringShape(sb, 0, new int[shape.length]);
        return sb.toString();
    }

    private void appendToStringShape(StringBuilder sb, int dim, int[] index) {
        if (dim == shape.length) {
            sb.append(get(index));
            return;
        }
        sb.append("[");
        for (int i = 0; i < shape[dim]; i++) {
            index[dim] = i;
            appendToStringShape(sb, dim + 1, index);
            if (i < shape[dim] - 1) sb.append(", ");
        }
        sb.append("]");
    }

    // Simple N-D index iterator over a given shape
    private static final class IndexIterator {
        private final int[] shape;
        private final int[] index;
        private boolean hasNext;

        IndexIterator(int[] shape) {
            this.shape = shape.clone();
            this.index = new int[shape.length];
            this.hasNext = product(shape) > 0;
            if (!hasNext) {
                // nothing to iterate
            }
        }

        boolean hasNext() {
            return hasNext;
        }

        int[] next() {
            if (!hasNext) {
                throw new IllegalStateException("Iterator exhausted");
            }
            int[] out = index.clone();
            // advance
            for (int d = shape.length - 1; d >= 0; d--) {
                index[d]++;
                if (index[d] < shape[d]) {
                    break;
                } else {
                    index[d] = 0;
                    if (d == 0) hasNext = false;
                }
            }
            return out;
        }
    }

    public List<?> toList() {
        return Arrays.asList(this.toArray());
    }
}