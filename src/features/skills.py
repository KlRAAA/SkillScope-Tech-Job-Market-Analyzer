"""Regex-based extraction of ~150 tech skills from job text.

Each skill has a canonical name, a category and a regex. Patterns are
case-insensitive unless the skill is listed in CASE_SENSITIVE (short or
ambiguous words such as "Go", "R", "React", "Swift").

Word boundaries: plain \\b fails for names that start or end with symbols
("C++", "C#", ".NET"), so every pattern is wrapped in custom lookarounds
that treat letters, digits, '+' and '#' as part of a word, plus a '.'
followed by a letter on the right (so "Node" does not match "Node.js").
"""

import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# (canonical name, category, pattern)
SKILLS: list[tuple[str, str, str]] = [
    # --- Languages -------------------------------------------------------------
    ("Python", "Language", r"python"),
    ("Java", "Language", r"java(?!\s*script)"),
    ("JavaScript", "Language", r"javascript|java script|ecmascript|es6"),
    ("TypeScript", "Language", r"typescript"),
    ("C++", "Language", r"c\+\+|cpp"),
    ("C#", "Language", r"c#|c sharp|csharp"),
    (
        "C",
        "Language",
        r"C(?=\s*/\s*C\+\+|\s*,\s*C\+\+|\s+(?:and|or)\s+C\+\+|\s+programming|\s+language)",
    ),
    (
        "Go",
        "Language",
        r"golang|Go(?=\s*(?:/|,|\)|\s+(?:and|or)\s+[A-Z]|\s+programming|\s+language))",
    ),
    ("Rust", "Language", r"Rust"),
    ("Ruby", "Language", r"ruby"),
    ("PHP", "Language", r"php"),
    ("Kotlin", "Language", r"kotlin"),
    ("Swift", "Language", r"Swift(?:UI)?"),
    ("Objective-C", "Language", r"objective[\s-]?c"),
    ("Scala", "Language", r"scala"),
    (
        "R",
        "Language",
        r"R(?=\s*(?:,|/|\)|\s+(?:and|or)\s+(?:Python|SQL|SAS)|\s+programming|\s+studio))|(?i:rstudio)",
    ),
    ("SQL", "Language", r"sql|t-sql|tsql|pl/sql|plsql"),
    ("Bash", "Language", r"bash|shell scripting|shell script"),
    ("PowerShell", "Language", r"powershell"),
    ("Perl", "Language", r"perl"),
    ("MATLAB", "Language", r"matlab"),
    ("SAS", "Language", r"SAS"),
    ("VBA", "Language", r"vba"),
    ("COBOL", "Language", r"cobol"),
    ("Dart", "Language", r"Dart"),
    ("Groovy", "Language", r"groovy"),
    ("HTML", "Language", r"html5?"),
    ("CSS", "Language", r"css3?"),
    # --- Frontend / mobile -----------------------------------------------------
    # "React quickly to ..." at the start of a sentence is the verb, not the library.
    (
        "React",
        "Frontend",
        (
            r"React(?:\.?js)?(?!\s*Native)"
            r"(?!\s+(?:to|quickly|promptly|swiftly|appropriately|accordingly)\b)"
            r"|(?i:reactjs)"
        ),
    ),
    ("Angular", "Frontend", r"angular(?:\.?js)?"),
    ("Vue.js", "Frontend", r"vue(?:\.?js)?"),
    ("Next.js", "Frontend", r"next\.?js"),
    ("Redux", "Frontend", r"redux"),
    ("jQuery", "Frontend", r"jquery"),
    ("Sass", "Frontend", r"sass|scss"),
    ("Tailwind", "Frontend", r"tailwind(?:\s*css)?"),
    ("Bootstrap", "Frontend", r"Bootstrap"),
    ("Webpack", "Frontend", r"webpack"),
    ("React Native", "Mobile", r"react[\s-]?native"),
    ("Flutter", "Mobile", r"flutter"),
    ("iOS", "Mobile", r"ios"),
    ("Android", "Mobile", r"android"),
    ("Xamarin", "Mobile", r"xamarin"),
    # --- Backend frameworks ----------------------------------------------------
    ("Node.js", "Backend", r"node\.?js"),
    ("Express", "Backend", r"express\.?js"),
    ("Django", "Backend", r"django"),
    ("Flask", "Backend", r"flask"),
    ("FastAPI", "Backend", r"fastapi"),
    (
        "Spring",
        "Backend",
        r"spring\s*boot|spring\s*framework|spring\s*mvc|spring\s*cloud",
    ),
    (".NET", "Backend", r"\.net(?:\s*core)?|dotnet"),
    ("ASP.NET", "Backend", r"asp\.net"),
    ("Ruby on Rails", "Backend", r"(?i:ruby on rails)|Rails"),
    ("Laravel", "Backend", r"laravel"),
    ("GraphQL", "Backend", r"graphql"),
    (
        "REST APIs",
        "Backend",
        r"rest(?:ful)?\s*(?:api|apis|services|web services)|restful",
    ),
    ("Microservices", "Backend", r"micro[\s-]?services?"),
    ("Kafka", "Backend", r"kafka"),
    ("RabbitMQ", "Backend", r"rabbitmq"),
    # --- Databases ---------------------------------------------------------------
    ("PostgreSQL", "Database", r"postgre(?:s|sql)"),
    ("MySQL", "Database", r"mysql"),
    ("SQL Server", "Database", r"sql server|mssql|ms sql"),
    ("Oracle", "Database", r"oracle"),
    ("MongoDB", "Database", r"mongo(?:db)?"),
    ("Redis", "Database", r"redis"),
    ("Cassandra", "Database", r"cassandra"),
    ("DynamoDB", "Database", r"dynamo\s*db"),
    ("Elasticsearch", "Database", r"elastic\s*search|elk stack"),
    ("NoSQL", "Database", r"nosql"),
    ("Snowflake", "Database", r"snowflake"),
    ("BigQuery", "Database", r"big\s*query"),
    ("Redshift", "Database", r"redshift"),
    # --- Cloud & DevOps ------------------------------------------------------------
    ("AWS", "Cloud", r"aws|amazon web services"),
    ("Azure", "Cloud", r"azure"),
    ("GCP", "Cloud", r"gcp|google cloud"),
    ("AWS Lambda", "Cloud", r"lambda"),
    ("Serverless", "Cloud", r"serverless"),
    ("Docker", "DevOps", r"docker"),
    ("Kubernetes", "DevOps", r"kubernetes|k8s"),
    ("OpenShift", "DevOps", r"openshift"),
    ("Helm", "DevOps", r"Helm"),
    ("Terraform", "DevOps", r"terraform"),
    ("CloudFormation", "DevOps", r"cloud\s*formation"),
    ("Ansible", "DevOps", r"ansible"),
    ("Puppet", "DevOps", r"Puppet"),
    ("Chef", "DevOps", r"Chef(?=\s*(?:,|/|\)|\s+(?:and|or)\s+[A-Z]))"),
    ("Jenkins", "DevOps", r"jenkins"),
    ("GitHub Actions", "DevOps", r"github actions"),
    (
        "CI/CD",
        "DevOps",
        r"ci\s*/\s*cd|continuous integration|continuous delivery|continuous deployment",
    ),
    ("Git", "DevOps", r"git"),
    ("GitHub", "DevOps", r"github"),
    ("GitLab", "DevOps", r"gitlab"),
    ("Bitbucket", "DevOps", r"bitbucket"),
    ("Azure DevOps", "DevOps", r"azure devops"),
    ("Linux", "DevOps", r"linux|red hat|rhel|ubuntu|centos"),
    ("Unix", "DevOps", r"unix"),
    ("Prometheus", "DevOps", r"prometheus"),
    ("Grafana", "DevOps", r"grafana"),
    ("Datadog", "DevOps", r"datadog"),
    ("Splunk", "DevOps", r"splunk"),
    # --- Data & ML -----------------------------------------------------------------
    ("Machine Learning", "Data/ML", r"machine learning|\bml\b"),
    ("Deep Learning", "Data/ML", r"deep learning|neural networks?"),
    ("NLP", "Data/ML", r"nlp|natural language processing"),
    ("Computer Vision", "Data/ML", r"computer vision"),
    ("LLMs", "Data/ML", r"llms?|large language models?"),
    ("Generative AI", "Data/ML", r"generative ai|gen\s*ai"),
    ("TensorFlow", "Data/ML", r"tensorflow"),
    ("PyTorch", "Data/ML", r"pytorch"),
    ("Keras", "Data/ML", r"keras"),
    ("scikit-learn", "Data/ML", r"scikit[\s-]?learn|sklearn"),
    ("Pandas", "Data/ML", r"pandas"),
    ("NumPy", "Data/ML", r"numpy"),
    ("Spark", "Data/ML", r"(?i:apache spark|pyspark)|Spark"),
    ("Hadoop", "Data/ML", r"hadoop|hdfs|hive"),
    ("Databricks", "Data/ML", r"databricks"),
    ("Airflow", "Data/ML", r"airflow"),
    ("dbt", "Data/ML", r"dbt"),
    ("ETL", "Data/ML", r"etl|elt"),
    ("Data Warehousing", "Data/ML", r"data warehous\w*"),
    ("Data Modeling", "Data/ML", r"data model(?:ing|ling)"),
    ("Statistics", "Data/ML", r"statistics|statistical"),
    ("A/B Testing", "Data/ML", r"a/b test\w*"),
    ("Tableau", "Data/ML", r"tableau"),
    ("Power BI", "Data/ML", r"power\s*bi"),
    ("Looker", "Data/ML", r"looker"),
    ("Excel", "Data/ML", r"Excel|(?i:ms excel|microsoft excel)"),
    # --- Testing ---------------------------------------------------------------------
    ("Selenium", "Testing", r"selenium"),
    ("Cypress", "Testing", r"cypress"),
    ("Jest", "Testing", r"Jest"),
    ("JUnit", "Testing", r"junit"),
    ("pytest", "Testing", r"pytest"),
    ("Postman", "Testing", r"postman"),
    ("Unit Testing", "Testing", r"unit test\w*"),
    ("Test Automation", "Testing", r"test automation|automated test\w*"),
    # --- Security & infrastructure -----------------------------------------------------
    ("Cybersecurity", "Security", r"cyber\s*security|information security|infosec"),
    ("SIEM", "Security", r"siem"),
    ("Penetration Testing", "Security", r"penetration test\w*|pen test\w*"),
    ("IAM", "Security", r"identity and access management|\biam\b"),
    ("Firewalls", "Security", r"firewalls?"),
    ("CISSP", "Security", r"cissp"),
    ("Security+", "Security", r"security\+"),
    (
        "Networking",
        "Infrastructure",
        r"tcp/ip|dns|dhcp|routing and switching|lan/wan|networking",
    ),
    ("Cisco", "Infrastructure", r"cisco|ccna|ccnp"),
    ("VMware", "Infrastructure", r"vmware|vsphere"),
    ("Active Directory", "Infrastructure", r"active directory"),
    ("Windows Server", "Infrastructure", r"windows server"),
    ("Microsoft 365", "Infrastructure", r"office 365|microsoft 365|o365|m365"),
    ("ITIL", "Infrastructure", r"itil"),
    ("ServiceNow", "Infrastructure", r"servicenow"),
    # --- Enterprise platforms ---------------------------------------------------------
    ("Salesforce", "Enterprise", r"salesforce"),
    ("SAP", "Enterprise", r"SAP"),
    ("SharePoint", "Enterprise", r"sharepoint"),
    ("Power Apps", "Enterprise", r"power\s*apps"),
    ("Mainframe", "Enterprise", r"mainframe|z/os"),
    # --- Embedded ------------------------------------------------------------------------
    ("Embedded Systems", "Embedded", r"embedded (?:systems?|software|c)"),
    ("RTOS", "Embedded", r"rtos"),
    ("FPGA", "Embedded", r"fpga"),
    ("Verilog/VHDL", "Embedded", r"verilog|vhdl"),
    # --- Practices & tools ---------------------------------------------------------------
    ("Agile", "Practice", r"agile"),
    ("Scrum", "Practice", r"scrum"),
    ("Jira", "Practice", r"jira"),
    ("Confluence", "Practice", r"confluence"),
    ("System Design", "Practice", r"system design|distributed systems"),
    ("Object-Oriented Programming", "Practice", r"object[\s-]oriented|\boop\b"),
    ("Data Structures & Algorithms", "Practice", r"data structures|algorithms"),
    ("Figma", "Practice", r"figma"),
    (
        "Unity",
        "Practice",
        r"Unity(?:3d)?(?=\s*(?:,|/|\)|\s+(?:and|or|engine|developer)))",
    ),
]

# Short or ambiguous names that must match with their exact capitalization.
CASE_SENSITIVE = {
    "C", "Go", "Rust", "Swift", "R", "SAS", "Dart", "React", "Bootstrap", "Helm",
    "Puppet", "Chef", "Spark", "Jest", "SAP", "Unity", "Excel", "Ruby on Rails",
}  # fmt: skip

SKILL_NAMES = [name for name, _, _ in SKILLS]
SKILL_CATEGORY = {name: category for name, category, _ in SKILLS}

_LEFT = r"(?<![\w+#])"
_RIGHT = r"(?![\w+#])(?!\.\w)"


def _compile(name: str, pattern: str) -> re.Pattern:
    flags = 0 if name in CASE_SENSITIVE else re.IGNORECASE
    return re.compile(f"{_LEFT}(?:{pattern}){_RIGHT}", flags)


_PATTERNS = {name: _compile(name, pattern) for name, _, pattern in SKILLS}


def skill_column(name: str) -> str:
    """Column-safe name, e.g. 'C++' -> 'skill_c_plus_plus'."""
    slug = (
        name.lower()
        .replace("++", "_plus_plus")
        .replace("#", "_sharp")
        .replace(".", "_dot_")
    )
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return f"skill_{slug}"


def extract_skills(text: str | None) -> list[str]:
    """Return the canonical names of all skills mentioned in the text."""
    if not isinstance(text, str) or not text:
        return []
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]


def _match_chunk(texts: list[str]) -> np.ndarray:
    matrix = np.zeros((len(texts), len(SKILLS)), dtype=np.uint8)
    for j, pattern in enumerate(_PATTERNS.values()):
        matrix[:, j] = [pattern.search(t) is not None for t in texts]
    return matrix


def skill_matrix(texts: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """0/1 column per skill plus a skill_count column, aligned to texts.index.

    ~166 regexes over long descriptions is CPU-bound, so rows are split into
    chunks and processed in parallel (n_jobs=-1 uses all cores).
    """
    values = texts.fillna("").tolist()
    n_chunks = max(1, min(len(values) // 500, 4 * (os.cpu_count() or 1)))
    chunks = [
        c.tolist() for c in np.array_split(np.array(values, dtype=object), n_chunks)
    ]
    parts = Parallel(n_jobs=n_jobs)(delayed(_match_chunk)(c) for c in chunks)
    matrix = np.vstack(parts) if parts else np.zeros((0, len(SKILLS)), dtype=np.uint8)
    df = pd.DataFrame(
        matrix, index=texts.index, columns=[skill_column(n) for n in SKILL_NAMES]
    )
    df["skill_count"] = matrix.sum(axis=1).astype(np.int16)
    return df
