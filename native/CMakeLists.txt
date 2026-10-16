cmake_minimum_required(VERSION 3.18)
project(membench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MEMBENCH_JNI "Build libmembench_jni.so for the Android app" OFF)
option(MEMBENCH_CLI "Build the membench command-line executable" ON)

# ---- glslc (ships with the NDK under shader-tools/) --------------------------------
if(ANDROID_NDK)
  set(_ndk "${ANDROID_NDK}")
elseif(DEFINED ENV{ANDROID_NDK_HOME})
  set(_ndk "$ENV{ANDROID_NDK_HOME}")
endif()
find_program(GLSLC glslc
  HINTS "${_ndk}/shader-tools/windows-x86_64" "${_ndk}/shader-tools/linux-x86_64"
        "${_ndk}/shader-tools/darwin-x86_64" "$ENV{VULKAN_SDK}/bin"
  NO_CMAKE_FIND_ROOT_PATH REQUIRED)

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN_DIR}")

# ---- Vulkan headers: NDK sysroot on Android. Desktop (test) builds copy only the Vulkan
#      headers out of the NDK sysroot, or use -DVULKAN_INCLUDE_DIR=<dir with vulkan/>. ----
if(NOT ANDROID AND NOT VULKAN_INCLUDE_DIR)
  file(GLOB _sysinc "${_ndk}/toolchains/llvm/prebuilt/*/sysroot/usr/include")
  list(GET _sysinc 0 _sysinc)
  foreach(d vulkan vk_video)
    if(EXISTS "${_sysinc}/${d}")
      file(COPY "${_sysinc}/${d}" DESTINATION "${GEN_DIR}/vkinc")
    endif()
  endforeach()
  set(VULKAN_INCLUDE_DIR "${GEN_DIR}/vkinc")
endif()

# ---- shaders -> SPIR-V (embedded as C arrays) + disassembly for inspection ----------
set(SPV_INCS "")
foreach(s read write copy verify gemm_tiled randmm_alu)
  set(src "${SHADER_DIR}/${s}.comp")
  add_custom_command(
    OUTPUT "${GEN_DIR}/${s}.spv.inc" "${GEN_DIR}/${s}.spvasm"
    COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -I "${SHADER_DIR}" -mfmt=num -o "${GEN_DIR}/${s}.spv.inc" "${src}"
    COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -I "${SHADER_DIR}" -S -o "${GEN_DIR}/${s}.spvasm" "${src}"
    DEPENDS "${src}" "${SHADER_DIR}/common.glsl"
    COMMENT "glslc ${s}.comp")
  list(APPEND SPV_INCS "${GEN_DIR}/${s}.spv.inc")
endforeach()

# ---- cooperative-matrix GEMM (boost load): GL_KHR_cooperative_matrix needs a newer
#      compiler than the NDK's glslc. With glslang (GLSLANG_DIR / VULKAN_SDK / PATH) the
#      shaders are compiled here, otherwise the committed shaders/prebuilt/*.spv.inc are
#      used. Target "update_prebuilt_spv" refreshes the committed copies. ---------------
find_program(GLSLANG NAMES glslang glslangValidator
  HINTS "$ENV{GLSLANG_DIR}" "$ENV{GLSLANG_DIR}/bin" "$ENV{VULKAN_SDK}/bin" NO_CMAKE_FIND_ROOT_PATH)
set(_copy_cmds "")
foreach(s gemm_coop_f32 gemm_coop_f16 randmm_coop_f32 randmm_coop_f16)
  set(src "${SHADER_DIR}/${s}.comp")
  if(GLSLANG)
    add_custom_command(
      OUTPUT "${GEN_DIR}/${s}.spv.inc"
      COMMAND "${GLSLANG}" -V --target-env vulkan1.2 --quiet "-I${SHADER_DIR}" -x -o "${GEN_DIR}/${s}.spv.inc" "${src}"
      DEPENDS "${src}" "${SHADER_DIR}/gemm_coop.glsl" "${SHADER_DIR}/randmm_coop.glsl" "${SHADER_DIR}/common.glsl"
      COMMENT "glslang ${s}.comp")
    list(APPEND _copy_cmds COMMAND ${CMAKE_COMMAND} -E copy "${GEN_DIR}/${s}.spv.inc" "${SHADER_DIR}/prebuilt/")
  else()
    configure_file("${SHADER_DIR}/prebuilt/${s}.spv.inc" "${GEN_DIR}/${s}.spv.inc" COPYONLY)
  endif()
  list(APPEND SPV_INCS "${GEN_DIR}/${s}.spv.inc")
endforeach()
if(GLSLANG)
  add_custom_target(update_prebuilt_spv ${_copy_cmds} DEPENDS ${SPV_INCS})
else()
  message(STATUS "glslang not found: using prebuilt cooperative-matrix SPIR-V")
endif()

add_custom_target(membench_shaders DEPENDS ${SPV_INCS})

# ---- core --------------------------------------------------------------------------
add_library(membench_core STATIC src/membench.cpp src/vk_loader.cpp src/cpu_mem.cpp)
add_dependencies(membench_core membench_shaders)
target_include_directories(membench_core PUBLIC src PRIVATE "${GEN_DIR}")
if(NOT ANDROID)
  target_include_directories(membench_core PUBLIC "${VULKAN_INCLUDE_DIR}")
endif()
find_package(Threads REQUIRED)
target_link_libraries(membench_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(membench_core PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

if(MEMBENCH_CLI)
  add_executable(membench src/main_cli.cpp)
  target_link_libraries(membench PRIVATE membench_core)
  if(MINGW)
    target_link_options(membench PRIVATE -static)
  endif()
  if(ANDROID)
    # NDK r27+ adds debug info to every build type; strip the release executable.
    target_link_options(membench PRIVATE $<$<CONFIG:Release>:-s>)
  endif()
endif()

if(MEMBENCH_JNI AND ANDROID)
  add_library(membench_jni SHARED src/jni_bridge.cpp)
  target_link_libraries(membench_jni PRIVATE membench_core log)
endif()
