from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from DbIndexing.models.setup import job_table,engine,SessionLocal,job_embeddings_table
from DbIndexing.models.job_postings import JobPostings #NECCESSARY DONT REMOVE THIS UNUSED IMPORT
from DbIndexing.models.skills import Skills #NECCESSARY DONT REMOVE THIS UNUSED IMPORT
from DbIndexing.models.job_embeddings import JobEmbeddings
from DbIndexing.models.job_similarity_matrix import JobSimilarityMatrix
from sqlalchemy import select,func
import io
import numpy as np


MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class JobSimilarityIndexer:
    def __init__(self):
        self.rows_data = None
        self.df = None
        self.model = MODEL
        self.similarity_matrix = None

    def retrieve_jobs_data(self):
        offset = self.determine_offset()
        query = select(job_table).order_by(job_table.c.id).offset(offset)

        with engine.connect() as conn:
            result  = conn.execute(query)
            rows = result.mappings().all()
            self.rows_data = rows
            

    def batch_encode_all_jobs(self):
        if not self.rows_data:
            return
        
        embeddings = []
        for row in self.rows_data:
            embedding,job_id = self.encode_job(row)

            if embedding is None or job_id is None:
                return
            
            try:
                self.store_embedding(job_id,embedding.cpu().tolist())
            except:
                continue
            
            embeddings.append(embedding)
            

        return embeddings
    
    def encode_job(self,row):
        combined_text = (
            str(row.get("title", "")) + " " +
            str(row.get("field", "")) + " " +
            str(row.get("responsibilities", "")) + " " +
            str(row.get("minimum_requirements", "")) + " " +
            str(row.get("company", "")) + " " +
            str(row.get("type", ""))
        )

        job_id = int(row.get("id",0))

        embedding = self.model.encode(combined_text,convert_to_tensor=True)

        return embedding,job_id


    def calculate_sim_matrix(self):
        old_embeddings = self.load_embeddings()
        new_embeddings = self.batch_encode_all_jobs()

        if new_embeddings is None and old_embeddings is None:
            return None
        elif old_embeddings is None and new_embeddings is not None:
            sim_matrix = cosine_similarity(new_embeddings)
        elif new_embeddings is None and old_embeddings is not None:
            sim_matrix = cosine_similarity(old_embeddings)
        else:
            sim_matrix  = cosine_similarity(
                new_embeddings,
                old_embeddings,
            )
        
        self.similarity_matrix = self.similarity_matrix

        return sim_matrix

    def store_embedding(self,job_id,embedding:list):
        try:
            with SessionLocal() as session:
                job_embedding_obj = JobEmbeddings(
                                        job_id=job_id,
                                        embedding=embedding
                                    )
                session.add(job_embedding_obj)
                session.commit()
        except Exception as e:
            print(f"[-] Error: {str(e)}")

    def load_embeddings(self):
        query = select(job_embeddings_table).order_by(job_embeddings_table.c.job_id)

        with engine.connect() as conn:
            results = conn.execute(query)
            rows = results.mappings().all()
        
        embeddings = [np.array(row["embedding"]) for row in rows]
        return embeddings


    def store_sim_matrix(self,sim_matrix):
        buffer = io.BytesIO()
        np.save(buffer, sim_matrix)
        buffer.seek(0)
        matrix_bytes = buffer.read()

        with SessionLocal() as session:
            obj = JobSimilarityMatrix(matrix=matrix_bytes)
            session.add(obj)
            session.commit()
    

    def load_sim_matrix(self):
        with SessionLocal() as session:
            # Get the most recent entry
            obj = session.query(JobSimilarityMatrix).order_by(JobSimilarityMatrix.created_at.desc()).first()
            if obj:
                buffer = io.BytesIO(obj.matrix)
                buffer.seek(0)
                self.similarity_matrix = np.load(buffer)

    def determine_offset(self):
        with SessionLocal() as session:
            max_job_id = session.query(func.max(job_embeddings_table.c.job_id)).scalar() or 0
            return int(max_job_id)
        
    def fetch_for_embeddings_not_calculated(self):
        with SessionLocal() as session:
            stmt = (
                select(JobPostings.__table__.c)
                .where(JobPostings.embedding == None)
            )

        results = session.execute(stmt).mappings().all()
        self.rows_data = results



#--------------------------------------PIPELINES-------------------------------------------------
def perfect_scenario():
    '''Where the data follows offset'''
    indexer = JobSimilarityIndexer()
    indexer.retrieve_jobs_data()
    sim_matrix = indexer.calculate_sim_matrix()
    indexer.store_sim_matrix(sim_matrix)

def calculate_for_missing_embeddings():
    '''Recalculate similarity index and the embeddings where the embeddings missed before'''
    indexer = JobSimilarityIndexer()
    indexer.fetch_for_embeddings_not_calculated()
    _ = indexer.batch_encode_all_jobs()

    # new instance
    indexer_2 = JobSimilarityIndexer()
    indexer_2.retrieve_jobs_data()
    sim_matrix = indexer_2.calculate_sim_matrix()
    indexer_2.store_sim_matrix(sim_matrix)

# indexing pipeline
if __name__ == "__main__":
    perfect_scenario()
