import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from DbIndexing.models.setup import job_table,engine,SessionLocal,job_skill_table
from DbIndexing.models.skills import Skills
from DbIndexing.models.job_postings import JobPostings
from DbIndexing.models.job_embeddings import JobEmbeddings
from sqlalchemy import select,func

NLP = spacy.load('en_core_web_lg')

EXTRACTOR = SkillExtractor(NLP,SKILL_DB,PhraseMatcher)

class SkillManager:
    def __init__(self):
        self.nlp = NLP
        self.extractor = EXTRACTOR


    def fetch_db_data(self):
        offset = int(self.determine_offset())
        query = select(job_table.c.id,job_table.c.minimum_requirements).where(job_table.c.minimum_requirements.isnot(None)).order_by(job_table.c.id).offset(offset)

        with engine.connect() as conn:
            result  = conn.execute(query)
            rows = result.fetchall()
            return rows
        
    def determine_offset(self):
        with SessionLocal() as session:
            max_job_id = session.query(func.max(job_skill_table.c.job_id)).scalar() or 0
            return max_job_id

    def extract_skills(self,val_string:str):
        if not val_string or not isinstance(val_string,str):
            return []
        
        val_string = val_string.strip()
        if len(val_string) < 3:
            return []
        
        try:
            annotations = self.extractor.annotate(val_string)
            skills = [item["doc_node_value"] for item in annotations["results"]["ngram_scored"]]

            return skills
        except:
            return []
    
    def skills_pipeline(self):
        rows_data = self.fetch_db_data()

        for row in rows_data:
            job_id = row[0]
            requirements_tx = row[1]
            
            skills = self.extract_skills(requirements_tx)
            with SessionLocal() as session:
                for skill in skills:
                    self.save_to_db(session,job_id,str(skill).strip())

    def save_to_db(self,session,job_id,skill_name):
        try:
            job = session.query(JobPostings).get(job_id)
            if not job:
                return
            
            skill = session.query(Skills).filter_by(name=skill_name).first()
            if not skill: # create skill entry if it doesnt exist on the db
                skill = Skills()
                skill.name = skill_name

            job.skills.append(skill) # link

            # save
            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Error: {str(e)}")




# RUN
if __name__ == "__main__":
    extractor_obj = SkillManager()
    extractor_obj.skills_pipeline()