use super::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::kernel;
use crate::tensor::{BF16, ElemType, F32, Layout};
use cuda_async::device_box::DeviceBox;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::os::fd::AsRawFd;

mod cuda_context {
    use cuda_core::{CudaContext, CudaStream};
    use std::sync::{Arc, OnceLock};

    static CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
    static STREAM: OnceLock<Arc<CudaStream>> = OnceLock::new();

    pub fn ctx() -> &'static Arc<CudaContext> {
        CTX.get_or_init(|| CudaContext::new(0).expect("Failed to create CUDA context"))
    }

    pub fn stream() -> &'static Arc<CudaStream> {
        STREAM.get_or_init(|| {
            let ctx = ctx();
            ctx.default_stream()
        })
    }
}
