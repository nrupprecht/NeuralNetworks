//
//  Stopwatch.h
//  Network
//
//  Created by Nathaniel Rupprecht on 7/1/15.
//  Copyright (c) 2015 Nathaniel Rupprecht. All rights reserved.
//

#ifndef Network_Stopwatch_h
#define Network_Stopwatch_h

#include <chrono>
using std::chrono::system_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// A stopwatch class for high precision timing
struct Stopwatch {
    Stopwatch() : timing(false) {};
    
    typedef system_clock clockType;
    
    void start() {
        startTime = clockType::now();
        timing = true;
    }
    
    void end() {
        endTime = clockType::now();
        timing = false;
    }
    
    double time() {
        clockType::time_point time = timing ? clockType::now() : endTime;
        duration<double> time_span = duration_cast<duration<double> >(time - startTime);
        return time_span.count();
    }
    
private:
    system_clock::time_point startTime;
    system_clock::time_point endTime;
    bool timing;
};

#endif
