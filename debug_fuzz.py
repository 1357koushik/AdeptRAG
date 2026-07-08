from rapidfuzz import fuzz, utils
print("Machine Learning vs Machine learning")
print("Ratio:", fuzz.ratio("Machine Learning", "Machine learning"))
print("Token Sort:", fuzz.token_sort_ratio("Machine Learning", "Machine learning"))
print("Token Sort with processor:", fuzz.token_sort_ratio("Machine Learning", "Machine learning", processor=utils.default_process))

print("\nElon Musk vs Musk, Elon")
print("Token Sort:", fuzz.token_sort_ratio("Elon Musk", "Musk, Elon"))
print("Token Sort with processor:", fuzz.token_sort_ratio("Elon Musk", "Musk, Elon", processor=utils.default_process))
