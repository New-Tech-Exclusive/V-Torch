module main

pub struct Tensor {
    pub mut:
    data: []f32 // flat memory
    shape: []int // dimensions
    strides: []int // how to index into flat memory

    requires_grad bool
    grad    &Tensor = unsafe { nil }
    parents []&Tensor
    op      string
}

fn compute_strides(shape []int) []int {
 mut s := []int{len: shape.len}
 mut acc := 1
 for i :=  shape.len - 1; i >= 0; i-- {
    s[i] = acc
    acc *= shape[i]
 }   
}