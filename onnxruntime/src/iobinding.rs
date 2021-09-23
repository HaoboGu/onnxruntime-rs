//! Module contains iobinding types

use std::ffi::CString;
use std::fmt::Debug;

use crate::{
    error::{call_ort, Result},
    session::Session,
    OrtError, TypeToTensorElementDataType,
};
use ndarray::Array;
use onnxruntime_sys as sys;
use sys::OrtSession;

/// IoBinding is a wrapper for onnx io binding
#[derive(Debug)]
pub struct IoBinding {
    iobinding_ptr: *mut sys::OrtIoBinding,
}

impl IoBinding {
    /// Create a new iobinding for an onnx session
    pub fn new(session_ptr: *mut OrtSession) -> Result<IoBinding> {
        // where onnxruntime will write the iobind to
        let mut iobinding_ptr: *mut sys::OrtIoBinding = std::ptr::null_mut();
        let iobinding_ptr_ptr: *mut *mut sys::OrtIoBinding = &mut iobinding_ptr;
        // Create io-binding
        unsafe { call_ort(|ort| ort.CreateIoBinding.unwrap()(session_ptr, iobinding_ptr_ptr)) }
            .unwrap();
        Ok(IoBinding { iobinding_ptr })
    }

    /// Bind input to iobinding
    pub fn bind_input<T, D>(&self, session: &Session, name: &str, mut input: Array<T, D>)
    where
        T: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        // First, create ort value from input
        let data_ptr: *mut std::ffi::c_void = input.as_mut_ptr() as *mut std::ffi::c_void;
        let shape: Vec<i64> = input.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = input.shape().len();
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;
        unsafe {
            call_ort(|ort| {
                ort.CreateTensorWithDataAsOrtValue.unwrap()(
                    session.memory_info.ptr,
                    data_ptr,
                    input.len() * std::mem::size_of::<T>(),
                    shape_ptr,
                    shape_len,
                    T::tensor_element_data_type().into(),
                    tensor_ptr_ptr,
                )
            })
        }
        .map_err(OrtError::CreateTensorWithData)
        .unwrap();

        // Bind input
        unsafe {
            // Bind input name to an ort Value ptr
            call_ort(|ort| {
                ort.BindInput.unwrap()(
                    self.iobinding_ptr,
                    CString::new(name).unwrap().into_raw() as *const i8,
                    tensor_ptr,
                )
            })
        }
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::GraphOptimizationLevel;
    use ndarray::arr0;

    use crate::environment::Environment;

    use super::*;

    #[test]
    fn test_create_io_binding() {
        let env = Environment::builder().with_name("env").build().unwrap();
        let session = env
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap()
            .with_model_from_file("D:\\Projects\\onnxruntime-rs\\gpt2.onnx")
            .unwrap();
        let array = arr0::<i32>(123);
        let iobinding = IoBinding::new(session.session_ptr).unwrap();
        iobinding.bind_input(&session, "input_ids", array);
    }
}
