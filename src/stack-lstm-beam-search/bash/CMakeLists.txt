INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_lstm lstm-parse.cc ${SHARED_SRC})
target_link_libraries(disf_stack_lstm cnn ${Boost_LIBRARIES})
