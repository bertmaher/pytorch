cmake_minimum_required(VERSION 3.5)
project(nnc)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -march=native -Werror -Wno-deprecated")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TEST_DIR ../../../../bin/)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build" FORCE)
endif()

set(ENABLE_LLVM ON CACHE BOOL "Enable LLVM")
find_package(LLVM)
if (NOT LLVM_FOUND)
  set(ENABLE_LLVM OFF)
endif(NOT LLVM_FOUND)

if (ENABLE_LLVM)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(-DENABLE_LLVM ${LLVM_DEFINITIONS})
endif (ENABLE_LLVM)

add_library(nnc
  expr.cpp
  function.cpp
  ir.cpp
  ir_visitor.cpp
  asmjit_codegen.cpp
  llvm_codegen.cpp
  llvm_jit.cpp
  types.cpp
  ir_printer.cpp
  ir_mutator.cpp
  schedule.cpp
  tensor.cpp
  )
target_link_libraries(nnc PUBLIC c10 asmjit)
set_source_files_properties(llvm_jit.cpp PROPERTIES COMPILE_FLAGS -fno-rtti)

if (LLVM_FOUND)
  llvm_map_components_to_libnames(LLVM_LINK_LIBS
  support core irreader analysis executionengine instcombine object orcJIT
  runtimedyld scalaropts transformutils native ipo orcjit)

  target_link_libraries(nnc PRIVATE ${LLVM_LINK_LIBS})
endif (LLVM_FOUND)

option(BUILD_TX_TESTS "Build the tensorexpr tests" ON)

if (BUILD_TX_TESTS)
  add_custom_target(cpptest)
  add_subdirectory(../../../../third_party/googletest/ googletest EXCLUDE_FROM_ALL)

  set(TEST_SRCS
    tests/asmjit_test.cpp
    tests/expr_test.cpp
    tests/llvm_test.cpp
    tests/type_test.cpp
    tests/ir_printer_test.cpp
    tests/schedule_test.cpp
    tests/aten_test.cpp
    tests/cuda_test.cpp
    )

  add_library(test_lib
    tests/padded_buffer.cpp
    )
  target_link_libraries(test_lib PUBLIC c10 gtest)

  foreach(test_path ${TEST_SRCS})
    get_filename_component(filename ${test_path} NAME)
    string(REPLACE ".cpp" "" test_exec ${filename})
    add_executable(${test_exec} ${test_path})
    add_dependencies(cpptest ${test_exec})
    target_link_libraries(${test_exec} test_lib nnc gtest_main gtest ${ASMJIT_DEPS})
    set_target_properties(${test_exec} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
    add_test(${test_exec} ${TEST_DIR}/${test_exec})
  endforeach()
endif()
