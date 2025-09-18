set(this avpjoin)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
        *.hpp *.cpp)

add_executable(${this} ${srcs})
target_link_libraries(${this} avpjoin_lib)
