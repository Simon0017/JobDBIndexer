from sqlalchemy.orm import relationship
from DbIndexing.models.setup import Base,job_embeddings_table

class JobEmbeddings(Base):
    __table__ = job_embeddings_table

    # Define relationship to JobPostings
    job = relationship(
        "JobPostings",
        back_populates="embedding",
        uselist=False  # one-to-one
    )