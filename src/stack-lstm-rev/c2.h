#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

namespace cpyp {

    class Corpus {
        //typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > Map;
// typedef std::unordered_map<unsigned,std::string, std::hash<std::string> > ReverseMap;
    public:
        bool USE_SPELLING = false;

        std::map<int, std::vector<unsigned>> correct_act_sent;
        std::map<int, std::vector<unsigned>> sentences_buffer;
        std::map<int, std::vector<unsigned>> sentences_gen;
        std::map<int, std::vector<unsigned>> sentencesPos_buffer;

        std::map<int, std::vector<unsigned>> correct_act_sentDev;
        std::map<int, std::vector<unsigned>> sentencesDev_buffer;
        std::map<int, std::vector<unsigned>> sentencesDev_gen;
        std::map<int, std::vector<unsigned>> sentencesPosDev_buffer;
        std::map<int, std::vector<std::string>> sentencesStrDev_buffer;
        std::map<int, std::vector<std::string>> sentencesStrDev;
        std::map<int, std::vector<std::string>> sentencesStrDev_gen;
        unsigned nsentencesDev;

        unsigned nsentences;
        unsigned nwords;
        unsigned nactions;
        unsigned npos;

//   unsigned nsentencestest;
//   unsigned nsentencesdev;
        int max;
        int maxPos;

        std::map<std::string, unsigned> wordsToInt;
        std::map<unsigned, std::string> intToWords;
        std::vector<std::string> actions;

        std::map<std::string, unsigned> posToInt;
        std::map<unsigned, std::string> intToPos;

        int maxChars;
        std::map<std::string, unsigned> charsToInt;
        std::map<unsigned, std::string> intToChars;

        // String literals
        static constexpr const char *UNK = "UNK";
        static constexpr const char *BAD0 = "<BAD0>";

/*  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;*/



    public:
        Corpus() {
            max = 0;
            maxPos = 0;
            maxChars = 0; //Miguel
        }


        inline unsigned UTF8Len(unsigned char x) {
            if (x < 0x80) return 1;
            else if ((x >> 5) == 0x06) return 2;
            else if ((x >> 4) == 0x0e) return 3;
            else if ((x >> 3) == 0x1e) return 4;
            else if ((x >> 2) == 0x3e) return 5;
            else if ((x >> 1) == 0x7e) return 6;
            else return 0;
        }


        inline void load_correct_actions(std::string file) {
//    std::map<int,std::vector<unsigned>> correct_act_sent;
            std::map<int, std::vector<unsigned>> sentences;
            std::map<int, std::vector<unsigned>> sentencesPos;
            std::map<int, std::vector<std::string>> action_string;

            std::ifstream actionsFile(file);
            //correct_act_sent=new vector<vector<unsigned>>();
            std::string lineS;

            int count = -1;
            int sentence = -1;
            bool initial = false;
            bool first = true;
            wordsToInt[Corpus::BAD0] = 0;
            intToWords[0] = Corpus::BAD0;
            wordsToInt[Corpus::UNK] = 1; // unknown symbol
            intToWords[1] = Corpus::UNK;
            assert(max == 0);
            assert(maxPos == 0);
            max = 2;
            maxPos = 1;

            charsToInt[BAD0] = 1;
            intToChars[1] = "BAD0";
            maxChars = 1;

            std::vector<unsigned> current_sent;
            std::vector<unsigned> current_sent_pos;
            std::vector<std::string> current_action;
            while (getline(actionsFile, lineS)) {
                //istringstream iss(line);
                //string lineS;
                //iss>>lineS;
                ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
                ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
                if (lineS.empty()) {
                    count = 0;
                    if (!first) {
                        sentences[sentence] = current_sent;
                        sentencesPos[sentence] = current_sent_pos;
                    }

                    sentence++;
                    nsentences = sentence;

                    initial = true;
                    current_sent.clear();
                    current_sent_pos.clear();
                } else if (count == 0) {
                    first = false;
                    //stack and buffer, for now, leave it like this.
                    count = 1;
                    if (initial) {
                        // the initial line in each sentence may look like:
                        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
                        // first, get rid of the square brackets.
                        //lineS = lineS.substr(3, lineS.size() - 4);
                        size_t posIndex = lineS.rfind(']');
                        lineS = lineS.substr(0, posIndex);
                        lineS = lineS.substr(3);
                        //std::cout<<lineS<<std::endl;
                        // read the initial line, token by token "the-det," "cat-noun," ...
                        std::istringstream iss(lineS);
                        do {
                            std::string word;
                            iss >> word;
                            if (word.size() == 0) { continue; }
                            // remove the trailing comma if need be.
                            if (word[word.size() - 1] == ',') {
                                word = word.substr(0, word.size() - 1);
                            }
                            // split the string (at '-') into word and POS tag.
                            posIndex = word.rfind('$');
                            if (posIndex == std::string::npos) {
                                std::cerr << "cant find the dash in '" << word << "'" << std::endl;
                            }
                            assert(posIndex != std::string::npos);
                            std::string pos = word.substr(posIndex + 1);
                            word = word.substr(0, posIndex);
//          posIndex = pos.rfind('\'');
//          assert(posIndex==pos.size()-1) ;
//          pos = pos.substr(0, posIndex);
//          posIndex = word.find('\'');
//          assert(posIndex==0) ;
//          word = word.substr(posIndex + 1);
                            // new POS tag
                            if (posToInt[pos] == 0) {
                                posToInt[pos] = maxPos;
                                intToPos[maxPos] = pos;
                                npos = maxPos;
                                maxPos++;
                            }

                            // new word
                            if (wordsToInt[word] == 0) {
                                wordsToInt[word] = max;
                                intToWords[max] = word;
                                nwords = max;
                                max++;

                                unsigned j = 0;
                                while (j < word.length()) {
                                    std::string wj = "";
                                    for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                                        wj += word[h];
                                    }
                                    if (charsToInt[wj] == 0) {
                                        charsToInt[wj] = maxChars;
                                        intToChars[maxChars] = wj;
                                        maxChars++;
                                    }
                                    j += UTF8Len(word[j]);
                                }
                            }

                            current_sent.push_back(wordsToInt[word]);
                            current_sent_pos.push_back(posToInt[pos]);
                        } while (iss);
                    }
                    initial = false;
                }
                else if (count == 1) {
                    int i = 0;
                    bool found = false;
                    std::vector<std::string> a_str = action_string[sentence];
                    a_str.push_back(lineS);
                    action_string[sentence] = a_str;
                    for (auto a: actions) {
                        if (a == lineS) {
                            std::vector<unsigned> a = correct_act_sent[sentence];
                            a.push_back(i);
                            correct_act_sent[sentence] = a;
                            found = true;
                        }
                        i++;
                    }
                    if (!found) {
                        actions.push_back(lineS);
                        std::vector<unsigned> a = correct_act_sent[sentence];
                        a.push_back(actions.size() - 1);
                        correct_act_sent[sentence] = a;
                    }
                    count = 0;
                }
            }

            // Add the last sentence.
            if (current_sent.size() > 0) {
                sentences[sentence] = current_sent;
                sentencesPos[sentence] = current_sent_pos;
                sentence++;
                nsentences = sentence;
                current_sent.clear();
                current_sent_pos.clear();
            }

            actionsFile.close();
/*	std::string oov="oov";
	posToInt[oov]=maxPos;
        intToPos[maxPos]=oov;
        npos=maxPos;
        maxPos++;
        wordsToInt[oov]=max;
        intToWords[max]=oov;
        nwords=max;
        max++;*/

            std::cerr << "done" << "\n";
            for (auto a: actions) {
                std::cerr << a << "\n";
            }
            nactions = actions.size();
            std::cerr << "nactions:" << nactions << "\n";
            std::cerr << "nwords:" << nwords << "\n";
            for (unsigned i = 0; i < npos; i++) {
                std::cerr << i << ":" << intToPos[i] << "\n";
            }
            nactions = actions.size();


            for (unsigned sii = 0; sii < nsentences; ++sii) {
                current_sent = sentences[sii];
                current_sent_pos = sentencesPos[sii];
                current_action = action_string[sii];
                unsigned cur = 0;
                for (unsigned m = 0; m < current_action.size(); ++m) {
                    if (current_action[m][0] == 'O') {
                        std::vector<unsigned> buf = sentences_buffer[sii];
                        buf.push_back(current_sent[cur]);
                        sentences_buffer[sii] = buf;
                        std::vector<unsigned> buf_pos = sentencesPos_buffer[sii];
                        buf_pos.push_back(current_sent_pos[cur]);
                        sentencesPos_buffer[sii] = buf_pos;
                        cur++;
                    }
                    else if (current_action[m][0] == 'S') {
                        std::vector<unsigned> gen = sentences_gen[sii];
                        gen.push_back(current_sent[cur]);
                        sentences_gen[sii] = gen;
                        cur++;
                    }
                }
            }
        }

        inline unsigned get_or_add_word(const std::string &word) {
            unsigned &id = wordsToInt[word];
            if (id == 0) {
                id = max;
                ++max;
                intToWords[id] = word;
                nwords = max;
            }
            return id;
        }

        inline void load_correct_actionsDev(std::string file) {

//            std::map<int, std::vector<unsigned>> correct_act_sentDev;
            std::map<int, std::vector<unsigned>> sentencesDev;
//            std::map<int, std::vector<unsigned>> sentencesDev_gen;
            std::map<int, std::vector<unsigned>> sentencesPosDev;
//            std::map<int, std::vector<std::string>> sentencesStrDev;
            std::map<int, std::vector<std::string>> action_string;
//            std::map<int, std::vector<std::string>> sentencesStrDev_gen;



            std::ifstream actionsFile(file);
            std::string lineS;

            assert(maxPos > 1);
            assert(max > 3);
            int count = -1;
            int sentence = -1;
            bool initial = false;
            bool first = true;
            std::vector<unsigned> current_sent;
            std::vector<unsigned> current_sent_pos;
            std::vector<std::string> current_sent_str;
            std::vector<std::string> current_action;
            while (getline(actionsFile, lineS)) {
                ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
                ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
                if (lineS.empty()) {
                    // an empty line marks the end of a sentence.
                    count = 0;
                    if (!first) {
                        sentencesDev[sentence] = current_sent;
                        sentencesPosDev[sentence] = current_sent_pos;
                        sentencesStrDev[sentence] = current_sent_str;
                    }

                    sentence++;
                    nsentencesDev = sentence;

                    initial = true;
                    current_sent.clear();
                    current_sent_pos.clear();
                    current_sent_str.clear();
                } else if (count == 0) {
                    first = false;
                    //stack and buffer, for now, leave it like this.
                    count = 1;
                    if (initial) {
                        // the initial line in each sentence may look like:
                        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
                        // first, get rid of the square brackets.
                        //lineS = lineS.substr(3, lineS.size() - 4);
                        size_t posIndex = lineS.rfind(']');
                        lineS = lineS.substr(0, posIndex);
                        lineS = lineS.substr(3);
                        // read the initial line, token by token "the-det," "cat-noun," ...
                        std::istringstream iss(lineS);
                        do {
                            std::string word;
                            iss >> word;
                            if (word.size() == 0) { continue; }
                            // remove the trailing comma if need be.
                            if (word[word.size() - 1] == ',') {
                                word = word.substr(0, word.size() - 1);
                            }
                            // split the string (at '-') into word and POS tag.
                            posIndex = word.rfind('$');
                            assert(posIndex != std::string::npos);
                            std::string pos = word.substr(posIndex + 1);
                            word = word.substr(0, posIndex);
                            if (posToInt[pos] == 0) {
                                posToInt[pos] = maxPos;
                                intToPos[maxPos] = pos;
                                npos = maxPos;
                                maxPos++;
                            }
                            // add an empty string for any token except OOVs (it is easy to
                            // recover the surface form of non-OOV using intToWords(id)).
                            current_sent_str.push_back(word);
                            // OOV word
                            if (wordsToInt[word] == 0) {
                                if (USE_SPELLING) {
                                    max = nwords + 1;
                                    //std::cerr<< "max:" << max << "\n";
                                    wordsToInt[word] = max;
                                    intToWords[max] = word;
                                    nwords = max;
                                } else {
                                    // save the surface form of this OOV before overwriting it.
                                    //current_sent_str[current_sent_str.size()-1] = word;
                                    word = Corpus::UNK;
                                }
                            }
                            current_sent.push_back(wordsToInt[word]);
                            current_sent_pos.push_back(posToInt[pos]);
                        } while (iss);
                    }
                    initial = false;
                } else if (count == 1) {
                    std::vector<std::string> a_str = action_string[sentence];
                    a_str.push_back(lineS);
                    action_string[sentence] = a_str;
                    auto actionIter = std::find(actions.begin(), actions.end(), lineS);
                    if (actionIter != actions.end()) {
                        unsigned actionIndex = std::distance(actions.begin(), actionIter);
                        correct_act_sentDev[sentence].push_back(actionIndex);
                    } else {
                        // TODO: right now, new actions which haven't been observed in training
                        // are not added to correct_act_sentDev. This may be a problem if the
                        // training data is little.
                    }
                    count = 0;
                }
            }

            // Add the last sentence.
            if (current_sent.size() > 0) {
                sentencesDev[sentence] = current_sent;
                sentencesPosDev[sentence] = current_sent_pos;
                sentencesStrDev[sentence] = current_sent_str;
                sentence++;
                nsentencesDev = sentence;
                current_sent.clear();
                current_sent_pos.clear();
                current_sent_str.clear();
            }

            actionsFile.close();

            for (unsigned sii = 0; sii < nsentencesDev; ++sii) {
                current_sent = sentencesDev[sii];
                current_sent_pos = sentencesPosDev[sii];
                current_sent_str = sentencesStrDev[sii];
                current_action = action_string[sii];
                unsigned cur = 0;
                unsigned cur_gen = 0;
                for (unsigned m = 0; m < current_action.size(); ++m) {
                    if (current_action[m][0] == 'O') {
                        std::vector<unsigned> buf = sentencesDev_buffer[sii];
                        buf.push_back(current_sent[cur]);
                        sentencesDev_buffer[sii] = buf;
                        std::vector<unsigned> buf_pos = sentencesPosDev_buffer[sii];
                        buf_pos.push_back(current_sent_pos[cur]);
                        sentencesPosDev_buffer[sii] = buf_pos;

                        std::vector<std::string> buf_str = sentencesStrDev_buffer[sii];
                        buf_str.push_back(current_sent_str[cur]);
                        sentencesStrDev_buffer[sii] = buf_str;
                        cur++;
                        cur_gen++;
                    }
                    else if (current_action[m][0] == 'S') {
                        std::vector<unsigned> gen = sentencesDev_gen[sii];
                        gen.push_back(current_sent[cur]);
                        sentencesDev_gen[sii] = gen;

                        std::vector<std::string> gen_str = sentencesStrDev_gen[sii];
                        gen_str.push_back(current_sent_str[cur] + "$" + std::to_string(cur_gen));
                        sentencesStrDev_gen[sii] = gen_str;
                        cur++;
                    }
                }
            }
        }

        void ReplaceStringInPlace(std::string &subject, const std::string &search,
                                  const std::string &replace) {
            size_t pos = 0;
            while ((pos = subject.find(search, pos)) != std::string::npos) {
                subject.replace(pos, search.length(), replace);
                pos += replace.length();
            }
        }

    };


} // namespace

#endif
