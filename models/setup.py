from sqlalchemy import create_engine,Table,MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL,echo=False)
Base = declarative_base()

SessionLocal = sessionmaker(bind=engine)

metadata = MetaData()

job_table = Table("job_postings", metadata, autoload_with=engine)
skills_table = Table("skills", metadata, autoload_with=engine)
job_skill_table = Table("job_skill", metadata, autoload_with=engine)


'''
 CREATE TABLE skills (
jobpostings(#     id SERIAL PRIMARY KEY,
jobpostings(#     name TEXT NOT NULL UNIQUE,
jobpostings(#     last_modified TIMESTAMPTZ DEFAULT NOW() NOT NULL
jobpostings(# );
CREATE TABLE
jobpostings=# \dt
            List of relations
 Schema |     Name     | Type  |  Owner
--------+--------------+-------+----------
 public | job_postings | table | postgres
 public | skills       | table | postgres
(2 rows)


jobpostings=# CREATE TABLE job_skill (
jobpostings(#     job_id INTEGER NOT NULL,
jobpostings(#     skill_id INTEGER NOT NULL,
jobpostings(#     PRIMARY KEY (job_id, skill_id),
jobpostings(#     FOREIGN KEY (job_id) REFERENCES job_postings(id) ON DELETE CASCADE,
jobpostings(#     FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
jobpostings(# );
CREATE TABLE
jobpostings=# \dt

'''