import pstats

#python .\proffileAnalysis 
#python -m cProfile -o profile_replay_optimizedv3.prof mainV.py
print("========== CUMULATIVE ==========")
pstats.Stats("antes_de_mejoras.prof").sort_stats("cumulative").print_stats(40)

print("\n========== TOTTIME ==========")
pstats.Stats("antes_de_mejoras.prof").sort_stats("tottime").print_stats(40)