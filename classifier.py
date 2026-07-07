import re
import math
import json
import string

def calculate_entropy(text: str) -> float:
    """Calculates Shannon entropy of the string in O(n)."""
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    length = len(text)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def classify_text(text: str) -> str:
    """
    Deterministically classifies text as Narrative Prose, Data Dump, or Uncertain
    based on O(n) weighted heuristics.
    """
    if not text or not text.strip():
        return json.dumps({"label": "Uncertain", "confidence": 0.99}, indent=2)

    lines = text.split('\n')
    num_lines = len(lines)
    length = len(text)

    # 1. Base Character & Token Metrics
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    punctuation = sum(1 for c in text if c in string.punctuation)
    whitespace = sum(1 for c in text if c.isspace())
    brackets = sum(1 for c in text if c in '()[]{}<>')
    
    # Ratios
    letter_ratio = letters / length
    digit_ratio = digits / length
    punct_ratio = punctuation / length
    bracket_ratio = brackets / length
    
    # 2. Line-based Heuristics
    blank_lines = sum(1 for line in lines if not line.strip())
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 40)
    non_empty_lines = num_lines - blank_lines if (num_lines - blank_lines) > 0 else 1
    
    short_line_ratio = short_lines / non_empty_lines
    avg_line_length = length / num_lines
    
    # 3. Regex Identifiers (Compiled for speed, scanning in O(n) across text)
    # Fast checks for common formats
    has_json = bool(re.search(r'^\s*[\{\[]\s*".*?"\s*:', text, re.DOTALL))
    has_xml_html = bool(re.search(r'<\??[a-zA-Z0-9]+.*?>.*?</[a-zA-Z0-9]+>', text, re.DOTALL))
    has_sql = bool(re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*?\b(FROM|INTO|TABLE)\b', text, re.IGNORECASE | re.DOTALL))
    has_md_table = bool(re.search(r'\|.*\|.*\n\s*\|[-:]+[-| :]*\|', text))
    has_md_code = bool(re.search(r'```[a-z]*\n.*?\n```', text, re.DOTALL))
    
    stack_trace_lines = len(re.findall(r'^\s*at\s+[\w\.]+\(|^\s*Traceback \(most recent call last\):|^\s*File ".*?\.py", line \d+', text, re.MULTILINE))
    kv_pairs = len(re.findall(r'^\s*"?[\w.-]+"?\s*[:=]\s*.*$', text, re.MULTILINE))
    kv_density = kv_pairs / non_empty_lines
    
    list_items = len(re.findall(r'^\s*[-*+]\s+|^\s*\d+\.\s+', text, re.MULTILINE))
    list_density = list_items / non_empty_lines
    
    # 4. Token & Word Level Heuristics
    # We approximate words/identifiers without heavy NLP
    tokens = re.findall(r'\b\w+\b', text)
    num_tokens = len(tokens) if len(tokens) > 0 else 1
    
    camel_snake_cases = sum(1 for t in tokens if re.match(r'^[a-z]+([A-Z][a-z0-9]+)+$|^[a-z]+(_[a-z0-9]+)+$', t))
    camel_snake_ratio = camel_snake_cases / num_tokens
    
    # Standard English Stopword approximation (very small fast O(1) lookup set)
    stopwords = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from"}
    stopword_count = sum(1 for t in tokens if t.lower() in stopwords)
    stopword_ratio = stopword_count / num_tokens
    
    # Specific tech patterns
    uuids = len(re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', text))
    ips = len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
    urls = len(re.findall(r'https?://[^\s]+', text))
    tech_pattern_density = (uuids + ips + urls) / non_empty_lines

    # Code Keyword Approximation
    code_keywords = {"def", "class", "import", "return", "if", "else", "elif", "for", "while", "try", "except", "function", "const", "let", "var", "public", "private", "struct", "void", "int", "bool", "self", "this", "True", "False", "None", "async", "await", "yield"}
    code_keyword_count = sum(1 for t in tokens if t in code_keywords)
    code_keyword_ratio = code_keyword_count / num_tokens

    # 5. Calculate Entropy
    entropy = calculate_entropy(text)

    # ----------------------------------------------------
    # Scoring Engine
    # ----------------------------------------------------
    prose_score = 0.0
    dump_score = 0.0
    code_score = 0.0
    
    # Narrative Prose Signals
    if letter_ratio > 0.7: prose_score += 20
    if 60 < avg_line_length < 150: prose_score += 15
    if bracket_ratio < 0.01: prose_score += 10
    if punct_ratio < 0.05: prose_score += 10
    if stopword_ratio > 0.15: prose_score += 30  # Strong indicator of natural language
    if camel_snake_ratio < 0.02: prose_score += 10
    
    # Data Dump Signals
    # Extremely strong indicators
    if has_json: dump_score += 60
    if has_sql: dump_score += 50
    if stack_trace_lines > 2: dump_score += 60
    if avg_line_length > 300: dump_score += 50  # Catches minified files, base64, etc.
    
    # Moderate indicators
    if kv_density > 0.3: dump_score += 40
    if list_density > 0.6: dump_score += 20  # Increased threshold so normal docs aren't penalized
    if bracket_ratio > 0.05: dump_score += 25
    if punct_ratio > 0.15: dump_score += 20
    if digit_ratio > 0.15: dump_score += 20
    if camel_snake_ratio > 0.1: dump_score += 30
    if short_line_ratio > 0.8: dump_score += 20
    if tech_pattern_density > 0.1: dump_score += 30
    
    # Source Code Signals
    if code_keyword_ratio > 0.03: code_score += 60
    if code_keyword_ratio > 0.08: code_score += 100
    if bracket_ratio > 0.02 and camel_snake_ratio > 0.05: code_score += 60
    if has_md_code: code_score += 40

    # Entropy modifiers (Random hashes/data or extremely repetitive logs)
    if entropy > 5.5 or entropy < 3.0: 
        dump_score += 15
        
    total = prose_score + dump_score + code_score
    if total == 0:
        return json.dumps({"label": "Uncertain", "confidence": 0.50}, indent=2)
    
    # Normalize Probabilities
    p_prose = prose_score / total
    p_dump = dump_score / total
    p_code = code_score / total
    
    max_p = max(p_prose, p_dump, p_code)
    
    # Decision boundaries enforcing "Near-Zero False Positives"
    if max_p < 0.50:
        return json.dumps({"label": "Uncertain", "confidence": round(max_p, 2)}, indent=2)
    elif p_code == max_p:
        return json.dumps({"label": "Source Code", "confidence": round(p_code, 2)}, indent=2)
    elif p_prose == max_p:
        return json.dumps({"label": "Narrative Prose", "confidence": round(p_prose, 2)}, indent=2)
    else:
        return json.dumps({"label": "Data Dump", "confidence": round(p_dump, 2)}, indent=2)

# --- Example Usage ---
if __name__ == "__main__":
    sample_prose = "Heuristics are strategies derived from previous experiences with similar problems. They rely on using readily accessible, though loosely applicable, information to control problem-solving in human beings and machines."
    sample_dump = '{"id": 1234, "name": "heuristics_engine", "status": "active", "metrics": {"cpu": 0.99, "mem": 0.45}}'
    
    print(classify_text(sample_prose))
    print(classify_text(sample_dump))