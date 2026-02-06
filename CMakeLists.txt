cmake_minimum_required(VERSION 3.28)
project(model_checker)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/minisat)
include_directories(target/cxxbridge)

add_executable(model_checker
        src/minisat/File.C
        src/minisat/File.h
        src/minisat/Global.h
        src/minisat/Heap.h
        src/minisat/Main.C
        src/minisat/Proof.C
        src/minisat/Proof.h
        src/minisat/Solver.C
        src/minisat/Solver.h
        src/minisat/SolverTypes.h
        src/minisat/Sort.h
        src/minisat/Test.C
        src/minisat/Test.h
        src/minisat/VarOrder.h
        src/minisat/Api.C
        src/minisat/Api.h)

add_compile_options(-Wno-deprecated)

find_package(ZLIB REQUIRED)
target_link_libraries(model_checker PRIVATE ZLIB::ZLIB)
