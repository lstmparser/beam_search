INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_lstm_gen lstm-parse.cc ${SHARED_SRC})
target_link_libraries(disf_stack_lstm_gen cnn ${Boost_LIBRARIES})
