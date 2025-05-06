#ifndef PROFILING_H
#define PROFILING_H

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <algorithm>

class Profiler {
private:
    struct ProfilePoint {
        std::string name;
        double duration;
        int calls;
    };

    static std::map<std::string, ProfilePoint> profile_points;
    static bool enabled;

public:
    static void enable() { enabled = true; }
    static void disable() { enabled = false; }
    static bool is_enabled() { return enabled; }

    static void record(const std::string& name, double duration) {
        if (!enabled) return;

        if (profile_points.find(name) == profile_points.end()) {
            profile_points[name] = {name, duration, 1};
        } else {
            profile_points[name].duration += duration;
            profile_points[name].calls++;
        }
    }

    static void print_results() {
        if (profile_points.empty()) {
            std::cout << "No profiling data collected." << std::endl;
            return;
        }

        // Convert map to vector for sorting
        std::vector<ProfilePoint> points;
        double total_time = 0.0;

        for (const auto& pair : profile_points) {
            points.push_back(pair.second);
            total_time += pair.second.duration;
        }

        // Sort by total duration (descending)
        std::sort(points.begin(), points.end(),
                 [](const ProfilePoint& a, const ProfilePoint& b) {
                     return a.duration > b.duration;
                 });

        // Print header
        std::cout << "\n--- Profiling Results ---" << std::endl;
        std::cout << std::left << std::setw(40) << "Section"
                  << std::right << std::setw(15) << "Time (s)"
                  << std::right << std::setw(15) << "Calls"
                  << std::right << std::setw(15) << "Avg Time (ms)"
                  << std::right << std::setw(15) << "% of Total" << std::endl;

        std::cout << std::string(100, '-') << std::endl;

        // Print each profile point
        for (const auto& point : points) {
            double percent = (point.duration / total_time) * 100.0;
            double avg_ms = (point.duration / point.calls) * 1000.0;

            std::cout << std::left << std::setw(40) << point.name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(6) << point.duration
                      << std::right << std::setw(15) << point.calls
                      << std::right << std::setw(15) << std::fixed << std::setprecision(3) << avg_ms
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << percent << "%" << std::endl;
        }

        std::cout << std::string(100, '-') << std::endl;
        std::cout << std::left << std::setw(40) << "Total"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(6) << total_time << std::endl;
    }

    static void reset() {
        profile_points.clear();
    }
};

// Static members are defined in the header but with inline to avoid multiple definition errors
inline std::map<std::string, Profiler::ProfilePoint> Profiler::profile_points;
inline bool Profiler::enabled = false;

// Macro for easy profiling
#define PROFILE_FUNCTION(name) \
    auto start_##name = std::chrono::high_resolution_clock::now(); \
    auto end_##name = std::chrono::high_resolution_clock::now(); \
    std::string profile_name_##name = #name; \
    class ScopedProfiler_##name { \
    public: \
        ScopedProfiler_##name(std::chrono::high_resolution_clock::time_point& start, \
                           std::chrono::high_resolution_clock::time_point& end, \
                           const std::string& name) \
            : start_(start), end_(end), name_(name) {} \
        ~ScopedProfiler_##name() { \
            end_ = std::chrono::high_resolution_clock::now(); \
            std::chrono::duration<double> elapsed = end_ - start_; \
            Profiler::record(name_, elapsed.count()); \
        } \
    private: \
        std::chrono::high_resolution_clock::time_point& start_; \
        std::chrono::high_resolution_clock::time_point& end_; \
        std::string name_; \
    }; \
    ScopedProfiler_##name profiler_##name(start_##name, end_##name, profile_name_##name);

#endif // PROFILING_H
