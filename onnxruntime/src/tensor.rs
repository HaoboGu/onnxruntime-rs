use std::ops::Deref;

use ndarray::Array;

use onnxruntime_sys as sys;

use crate::{
    error::status_to_result, g_ort, memory::MemoryInfo, OrtError, Result,
    TypeToTensorElementDataType,
};

// https://docs.rs/ndarray/0.13.1/ndarray/type.ArrayView.html#method.from_shape_ptr
pub struct Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType,
    D: ndarray::Dimension,
{
    pub(crate) c_ptr: *mut sys::OrtValue,
    array: Array<T, D>,
    memory_info: &'t MemoryInfo,
}

impl<'t, T, D> Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType,
    D: ndarray::Dimension,
{
    pub(crate) fn from_array<'m>(
        memory_info: &'m MemoryInfo,
        mut array: Array<T, D>,
    ) -> Result<Tensor<'t, T, D>>
    where
        'm: 't, // 'm outlives 't
    {
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;
        let tensor_values_ptr: *mut std::ffi::c_void = array.as_mut_ptr() as *mut std::ffi::c_void;
        assert_ne!(tensor_values_ptr, std::ptr::null_mut());

        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = array.shape().len() as u64;

        let status = unsafe {
            (*g_ort()).CreateTensorWithDataAsOrtValue.unwrap()(
                memory_info.ptr,
                tensor_values_ptr,
                (array.len() * std::mem::size_of::<T>()) as u64,
                shape_ptr,
                shape_len,
                T::tensor_element_data_type() as u32,
                tensor_ptr_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateTensorWithData)?;
        assert_ne!(tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { (*g_ort()).IsTensor.unwrap()(tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_eq!(is_tensor, 1);

        Ok(Tensor {
            c_ptr: tensor_ptr,
            array,
            memory_info,
        })
    }
}

impl<'t, T, D> Deref for Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType,
    D: ndarray::Dimension,
{
    type Target = Array<T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<'t, T, D> Drop for Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType,
    D: ndarray::Dimension,
{
    fn drop(&mut self) {
        // We need to let the C part free
        println!("Dropping Tensor.");
        unsafe { (*g_ort()).ReleaseValue.unwrap()(self.c_ptr) }

        self.c_ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AllocatorType, MemType};
    use ndarray::{arr0, arr1, arr2, arr3};

    #[test]
    fn tensor_from_array_0d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr0::<i32>(123);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[]);
    }

    #[test]
    fn tensor_from_array_1d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[6]);
    }

    #[test]
    fn tensor_from_array_2d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[2, 6]);
    }

    #[test]
    fn tensor_from_array_3d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 6]);
    }
}