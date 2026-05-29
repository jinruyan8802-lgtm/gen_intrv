import re
from typing import Dict

# 全大写 SQL 关键字 → 小写（TTS 引擎能自然读出单词而非逐字母拼读）
_SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
    "INSERT INTO", "UPDATE", "DELETE FROM", "CREATE TABLE", "ALTER TABLE",
    "DROP TABLE", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
    "OUTER JOIN", "CROSS JOIN", "ON", "AND", "OR", "NOT", "IN",
    "EXISTS", "BETWEEN", "LIKE", "IS NULL", "IS NOT NULL",
    "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN",
    "UNION", "UNION ALL", "LIMIT", "OFFSET", "ASC", "DESC",
    "SET", "VALUES", "INTO", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
    "PRIMARY KEY", "FOREIGN KEY", "INDEX", "UNIQUE", "DEFAULT",
    "AUTO_INCREMENT", "NOT NULL",
]

# SQL 关键字替换表（全大写 → 小写）
_SQL_REPLACEMENTS: Dict[str, str] = {}

def _build_sql_replacements():
    for kw in sorted(_SQL_KEYWORDS, key=len, reverse=True):
        _SQL_REPLACEMENTS[kw] = kw.lower()

_build_sql_replacements()

# 专有名词音标替换（TTS 引擎读不准的词）
_PROPER_NOUNS: Dict[str, str] = {
    "MySQL": "买-SQL",       # 常见发音，比逐字母读自然
    "PostgreSQL": "Postgres",
    "SQLite": "SQL-lite",
    "Redis": "瑞迪斯",
    "Nginx": "Engine-X",
    "Docker": "达克",
    "Kubernetes": "酷伯奈提斯",
    "K8s": "K八S",
    "Linux": "里纳克斯",
    "GitHub": "吉特哈布",
    "GraphQL": "Graph-Q-L",
    "gRPC": "g-R-P-C",
    "RESTful": "REST-ful",
    "JSON": "杰森",
    "JWT": "J-W-T",
    "OAuth": "O-Auth",
    "HTTPS": "H-T-T-P-S",
    "UUID": "U-U-I-D",
    "EOF": "E-O-F",
    "IO": "I-O",
    "CPU": "C-P-U",
    "GPU": "G-P-U",
    "SSD": "S-S-D",
    "RAM": "R-A-M",
    "QPS": "Q-P-S",
    "TPS": "T-P-S",
    "B+Tree": "B加Tree",
    "B-Tree": "B-Tree",
    "InnoDB": "In-no-D-B",
    "MyISAM": "My-I-SAM",
    "MVCC": "M-V-C-C",
    "WAL": "W-A-L",
    "LSM": "L-S-M",
    "CAP": "C-A-P",
    "ACID": "A-C-I-D",
    "Raft": "拉夫特",
    "Paxos": "帕克索斯",
    "ZooKeeper": "Zoo-Keeper",
    "Kafka": "卡夫卡",
    "Elasticsearch": "Elastic-search",
    "MongoDB": "Mongo-D-B",
    "Cassandra": "卡桑德拉",
    "HBase": "H-Base",
    "Hadoop": "哈杜普",
    "Spark": "斯帕克",
    "Flink": "弗林克",
    "TensorFlow": "Tensor-Flow",
    "PyTorch": "Py-Torch",
    "JVM": "J-V-M",
    "JIT": "J-I-T",
    "GC": "G-C",
    "CMS": "C-M-S",
    "G1": "G一",
    "ZGC": "Z-G-C",
    "DNS": "D-N-S",
    "CDN": "C-D-N",
    "VPN": "V-P-N",
    "TCP": "T-C-P",
    "UDP": "U-D-P",
    "HTTP": "H-T-T-P",
    "WebSocket": "Web-Socket",
    "Netty": "内提",
    "Thrift": "斯瑞夫特",
    "Protobuf": "Proto-buf",
    "SpringBoot": "Spring-Boot",
    "HashMap": "Hash-Map",
    "LinkedList": "Linked-List",
    "ArrayList": "Array-List",
    "ThreadPool": "Thread-Pool",
    "Reactor": "瑞艾克特",
    "Vuex": "V-uex",
    "Webpack": "Web-pack",
    "Vite": "维特",
    "TypeScript": "Type-Script",
    "Next.js": "Next-J-S",
    "Vue.js": "V-u-e-J-S",
    "Node.js": "Node-J-S",
}

# 按长度降序排列，确保长词先匹配
_sorted_nouns = sorted(_PROPER_NOUNS.items(), key=lambda x: len(x[0]), reverse=True)
_noun_patterns = [(re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v) for k, v in _sorted_nouns]

# SQL 关键字也需要按长度降序
_sql_pattern_pairs = sorted(_SQL_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True)
_sql_patterns = [(re.compile(r'\b' + re.escape(k) + r'\b'), v) for k, v in _sql_pattern_pairs]


def normalize_for_tts(text: str) -> str:
    """Normalize text for better TTS pronunciation in Chinese context.

    Rules:
    1. SQL keywords in ALL CAPS → lowercase (TTS reads words naturally vs spelling letters)
    2. Technical proper nouns → phonetic replacements
    3. Preserves surrounding punctuation and spacing
    """
    if not text:
        return text

    # Step 1: Replace proper nouns (higher priority, do first)
    for pattern, replacement in _noun_patterns:
        text = pattern.sub(replacement, text)

    # Step 2: Lowercase SQL keywords
    for pattern, replacement in _sql_patterns:
        text = pattern.sub(replacement, text)

    return text
