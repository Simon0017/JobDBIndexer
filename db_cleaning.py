'''This script provides class and objects
to clean the db for things like expired jobs_postings
'''

from DbIndexing.models.setup import SessionLocal,job_table,job_skill_table
from DbIndexing.models.job_postings import JobPostings
from DbIndexing.models.job_embeddings import JobEmbeddings
from DbIndexing.models.job_similarity_matrix import JobSimilarityMatrix
from DbIndexing.models.skills import Skills
from sqlalchemy import func, select,delete,update,cast,DateTime
from datetime import datetime, timedelta

class DbCleaner:
    '''Cleans the db for expired job postings'''

    def __init__(self):
        pass
    
    def delete_expired_jobs(self):
        with SessionLocal() as session:
            stmt = delete(job_table).where(
                job_table.c.application_deadline.is_not(None),
                cast(job_table.c.application_deadline, DateTime) < datetime.now()
            )
            session.execute(stmt)
            session.commit()
    
    def delete_low_quality_jobs(self):
        pass

    def delete_duplicate_jobs(self):
        """Keep the first (lowest id) row for every title+location and delete the rest."""
        with SessionLocal() as session:
            # Find duplicate groups and mark the extra rows
            duplicates_cte = (
                select(
                    job_table.c.id,
                    func.row_number()
                        .over(partition_by=(job_table.c.title, job_table.c.location), order_by=job_table.c.id)
                        .label('row_num')
                )
                .cte('duplicates')
            )

            stmt = (
                delete(job_table)
                .where(job_table.c.id.in_(
                    select(duplicates_cte.c.id).where(duplicates_cte.c.row_num > 1)
                ))
            )

            session.execute(stmt)
            session.commit()

    def delete_old_similarity_matrix(self):
        # a weeks old similarity matrix is not useful so we delete it
        with SessionLocal() as session:
            stmt = delete(JobSimilarityMatrix).where(cast(JobSimilarityMatrix.created_at, DateTime) < datetime.now() - timedelta(days=7))
            session.execute(stmt)
            session.commit()

    def delete_orphaned_relationships(self):
        with SessionLocal() as session:
            # delete orphaned job embeddings
            stmt = delete(JobEmbeddings).where(~JobEmbeddings.job_id.in_(select(job_table.c.id)))
            session.execute(stmt)
            session.commit()

            # delete orphaned job_skill m2m relationships
            stmt = delete(job_skill_table).where(~job_skill_table.c.job_id.in_(select(job_table.c.id)))
            session.execute(stmt)
            session.commit()


    def delete_old_job_postings(self):
        with SessionLocal() as session:
            stmt = delete(job_table).where(
                job_table.c.crawled_at.is_not(None),
                cast(job_table.c.crawled_at, DateTime) < datetime.now() - timedelta(days=45)
            )
            session.execute(stmt)
            session.commit()
    
    def clean_long_locations(self):
        # update the locations to thefirst 50 caharcters
        with SessionLocal() as session:
            stmt = (
                update(job_table)
                .where(func.length(job_table.c.location) > 50)
                .values(location=func.substr(job_table.c.location, 1, 50))
            )
            session.execute(stmt)
            session.commit()



if __name__ == "__main__":
    cleaner = DbCleaner()
    print("Cleaning the database...")
    print("Deleting old job postings...")
    cleaner.delete_old_job_postings()
    print("Deleting expired job postings...")
    cleaner.delete_expired_jobs()
    print("Deleting duplicate job postings...")
    cleaner.delete_duplicate_jobs()
    print("Cleaning long location names...")
    cleaner.clean_long_locations()
    print("Deleting old similarity matrix...")
    cleaner.delete_old_similarity_matrix()
    print("Deleting orphaned relationships...")
    cleaner.delete_orphaned_relationships()
    print("Database cleaning completed.")