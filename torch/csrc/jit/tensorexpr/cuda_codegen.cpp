#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"

#define DEBUG_PRINT 0

namespace torch {
namespace jit {
namespace tensorexpr {

// A RAII wrapper to manage a variable and name pair in the look-up table.
// TODO: move this to a more shared place.
class ScopedVarName {
 public:
  ScopedVarName(
      VarNameMap* mapping,
      const Variable* var,
      const std::string& name)
      : mapping_(mapping), var_(var) {
    auto iter = mapping->find(var);
    if (iter != mapping->end()) {
      throw std::runtime_error("Duplicate var entry: " + var->name_hint());
    }
    mapping->insert(std::make_pair(var, name));
  }

  ScopedVarName(
      UniqueNameManager* manager,
      const Variable* var,
      const std::string& name)
      : ScopedVarName(&manager->unique_name_mapping_, var, name) {}

  ~ScopedVarName() {
    auto iter = mapping_->find(var_);
    if (iter == mapping_->end()) {
      throw std::runtime_error("Invalid var entry: " + var_->name_hint());
    }
    mapping_->erase(var_);
  }

 private:
  ScopedVarName(const ScopedVarName&) = delete;
  ScopedVarName& operator=(const ScopedVarName&) = delete;

  VarNameMap* mapping_ = nullptr;
  const Variable* var_ = nullptr;
};

static int as_int(const Expr& expr) {
  const IntImm* v = expr.AsNode<IntImm>();
  return v->value();
}

static bool is_zero(const Expr& expr) {
  return as_int(expr) == 0;
}

static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static void getMajorMinor(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  AT_CUDA_NVRTC_CHECK(
      nvrtc().nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  AT_ASSERT(nvrtc_version.first >= 6);

  CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  CudaVersion max_dev_version(dev_version);
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  }
  if (dev_version > max_dev_version) {
    dev_version = max_dev_version;
  }
  major = dev_version.first;
  minor = dev_version.second;
}

void CudaPrinter::visit(const For* v) {
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    ScopedVarName var_name(
        name_manager(), v->var().node(), loop_options.gpu_block_index_str());
    v->body().accept(this);
    int gpu_block_index = loop_options.gpu_block_index();
    if (gpu_block_extents_.size() <= gpu_block_index) {
      gpu_block_extents_.resize(gpu_block_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }
    gpu_block_extents_[gpu_block_index] = v->stop();
  } else if (loop_options.is_gpu_thread_index()) {
    ScopedVarName var_name(
        name_manager(), v->var().node(), loop_options.gpu_thread_index_str());
    v->body().accept(this);
    int gpu_thread_index = loop_options.gpu_thread_index();
    if (gpu_thread_extents_.size() <= gpu_thread_index) {
      gpu_thread_extents_.resize(gpu_thread_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }
    gpu_thread_extents_[gpu_thread_index] = v->stop();
  } else {
    IRPrinter::visit(v);
  }
}

void CudaCodeGen::Initialize() {
  printer_.reset(new CudaPrinter(&oss_));
  // TODO: handle multiple kernels.
  // TODO: handle dynamic dimension.
  // TODO: call nvrtc.
  oss_ << "extern \"C\" __global__" << std::endl << "void f(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (int i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      oss_ << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const Var& var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();
    oss_ << dtype.ToCppString() << "* " << name_manager()->get_unique_name(var);
  }
  oss_ << ") {";

  oss_ << std::endl;
  ir_node()->accept(printer_.get());
  oss_ << std::endl;
  oss_ << "}";

  // Check that all block extents had been set.
  const std::vector<Expr>& gpu_block_extents = printer_->gpu_block_extents();
  const std::vector<Expr>& gpu_thread_extents = printer_->gpu_thread_extents();
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    if (gpu_block_extents[i].empty()) {
      throw std::runtime_error("Missing gpu_block_index: " + std::to_string(i));
    }
  }

#if DEBUG_PRINT
  std::cout << "stmt: " << std::endl;
  std::cout << oss_.str() << std::endl;
  std::cout << "block(";
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << gpu_block_extents[i];
  }
  std::cout << "), thread(";
  for (int i = 0; i < gpu_thread_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << gpu_thread_extents[i];
  }
  std::cout << ")" << std::endl;
  ;
#endif

  CompileToNVRTC(oss_.str());
}

void CudaCodeGen::call(const std::vector<CallArg>& args) {
  CHECK_EQ(args.size(), buffer_args().size());

  // TODO: move as much of this into the constructors.
  // TODO: handle dynamic shapes.
  const std::vector<Expr>& gpu_block_extents = printer_->gpu_block_extents();
  const std::vector<Expr>& gpu_thread_extents = printer_->gpu_thread_extents();
  CHECK(gpu_block_extents.size() <= 3);
  CHECK(gpu_thread_extents.size() <= 3);
  std::vector<int> gpu_block_extents_v(3, 1);
  std::vector<int> gpu_thread_extents_v(3, 1);
  // evaluate all the block/thread extents into values
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    gpu_block_extents_v[i] = as_int(gpu_block_extents[i]);
  }
  for (int i = 0; i < gpu_thread_extents.size(); i++) {
    gpu_thread_extents_v[i] = as_int(gpu_thread_extents[i]);
  }

  // Bind the buffer addresses into arguments
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  std::vector<void*> args_data(buffer_args.size());
  std::vector<void*> ptr_to_args(buffer_args.size());
  for (int i = 0; i < buffer_args.size(); i++) {
    args_data[i] = args[i].data();
    ptr_to_args[i] = &args_data[i];
  }

  // Launch the kernels
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      function_,
      gpu_block_extents_v[0],
      gpu_block_extents_v[1],
      gpu_block_extents_v[2],
      gpu_thread_extents_v[0],
      gpu_thread_extents_v[1],
      gpu_thread_extents_v[2],
      0,
      stream,
      ptr_to_args.data(),
      nullptr));
}

void CudaCodeGen::CompileToNVRTC(const std::string& code) {
  // Initializes driver's API context (if necessary)
  CUdevice device = 0;
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }

  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device);

  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);

#if DEBUG_PRINT
  std::cout << "major: " << major << ", "
            << "minor: " << minor << std::endl;
#endif

  // Creates the NVRTC program
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#ifdef __HIP_PLATFORM_HCC__
  std::vector<const char*> args = {};
#else
  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};
#endif

  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLogSize(program, &logsize));
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLog(program, log.data()));
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  CUmodule module;
  std::string name = "f";
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module, name.c_str()));
}

RegisterCodeGen<CudaCodeGen> reg("cuda_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
