import pstats

# Load the profiling results
p = pstats.Stats('profile_results.prof')

# Sort the results by cumulative time and print the top 10 functions
p.sort_stats('cumulative').print_stats(100)

# Optionally, you can sort by other criteria such as 'time' or 'calls'
# p.sort_stats('time').print_stats(10)
# p.sort_stats('calls').print_stats(10)
