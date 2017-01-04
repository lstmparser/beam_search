INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_lstm_newgen lstm-parse.cc ${SHARED_SRC})
target_link_libraries(disf_stack_lstm_newgen cnn ${Boost_LIBRARIES})
