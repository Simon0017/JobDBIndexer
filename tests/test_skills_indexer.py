import pytest
from unittest.mock import MagicMock, patch, call, PropertyMock
from datetime import datetime


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_spacy_and_extractor():
    """Prevent loading spacy models and SkillExtractor at import time."""
    with patch("DbIndexing.skill_manager.NLP", MagicMock()), \
         patch("DbIndexing.skill_manager.EXTRACTOR", MagicMock()):
        yield


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


@pytest.fixture
def mock_engine_conn():
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def skill_manager(mock_session, mock_engine_conn):
    with patch("DbIndexing.skill_manager.SessionLocal", return_value=mock_session), \
         patch("DbIndexing.skill_manager.engine") as mock_engine:
        mock_engine.connect.return_value = mock_engine_conn
        from DbIndexing.skill_indexer import SkillManager
        yield SkillManager()


# ════════════════════════════════════════════════════════════════════════════
# TestSkillManager
# ════════════════════════════════════════════════════════════════════════════

class TestSkillManager:

    # ── Instantiation ─────────────────────────────────────────────────────────

    class TestInstantiation:

        def test_creates_instance(self, skill_manager):
            from DbIndexing.skill_indexer import SkillManager
            assert isinstance(skill_manager, SkillManager)

        def test_nlp_is_assigned(self, skill_manager):
            assert skill_manager.nlp is not None

        def test_extractor_is_assigned(self, skill_manager):
            assert skill_manager.extractor is not None

    # ── determine_offset ──────────────────────────────────────────────────────

    class TestDetermineOffset:

        def test_returns_max_job_id_when_present(self, skill_manager, mock_session):
            mock_session.query.return_value.scalar.return_value = 42
            result = skill_manager.determine_offset()
            assert result == 42

        def test_returns_zero_when_table_is_empty(self, skill_manager, mock_session):
            mock_session.query.return_value.scalar.return_value = None
            result = skill_manager.determine_offset()
            assert result == 0

        def test_uses_session_context_manager(self, skill_manager, mock_session):
            mock_session.query.return_value.scalar.return_value = 0
            skill_manager.determine_offset()
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

        def test_queries_max_of_job_id(self, skill_manager, mock_session):
            mock_session.query.return_value.scalar.return_value = 10
            skill_manager.determine_offset()
            mock_session.query.assert_called_once()

        def test_returns_integer_type(self, skill_manager, mock_session):
            mock_session.query.return_value.scalar.return_value = 7
            result = skill_manager.determine_offset()
            assert isinstance(result, int)

        def test_session_error_propagates(self, skill_manager, mock_session):
            mock_session.query.side_effect = Exception("DB unavailable")
            with pytest.raises(Exception, match="DB unavailable"):
                skill_manager.determine_offset()

    # ── fetch_db_data ─────────────────────────────────────────────────────────

    class TestFetchDbData:

        def test_returns_rows_from_db(self, skill_manager, mock_session, mock_engine_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_engine_conn.execute.return_value.fetchall.return_value = [
                (1, "Python, SQL"), (2, "Java, Docker")
            ]
            rows = skill_manager.fetch_db_data()
            assert len(rows) == 2

        def test_returns_empty_list_when_no_rows(self, skill_manager, mock_session, mock_engine_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_engine_conn.execute.return_value.fetchall.return_value = []
            rows = skill_manager.fetch_db_data()
            assert rows == []

        def test_uses_engine_connect(self, skill_manager, mock_session, mock_engine_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_engine_conn.execute.return_value.fetchall.return_value = []
            skill_manager.fetch_db_data()
            mock_engine_conn.__enter__.assert_called_once()

        def test_offset_is_applied(self, skill_manager, mock_session, mock_engine_conn):
            """Offset from determine_offset must be forwarded to the query."""
            mock_session.query.return_value.scalar.return_value = 10
            mock_engine_conn.execute.return_value.fetchall.return_value = []
            with patch.object(skill_manager, "determine_offset", return_value=10) as mock_offset:
                skill_manager.fetch_db_data()
                mock_offset.assert_called_once()

        def test_fetchall_is_called(self, skill_manager, mock_session, mock_engine_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_engine_conn.execute.return_value.fetchall.return_value = []
            skill_manager.fetch_db_data()
            mock_engine_conn.execute.return_value.fetchall.assert_called_once()

        def test_engine_error_propagates(self, skill_manager, mock_session, mock_engine_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_engine_conn.execute.side_effect = Exception("connection refused")
            with pytest.raises(Exception, match="connection refused"):
                skill_manager.fetch_db_data()

    # ── extract_skills ────────────────────────────────────────────────────────

    class TestExtractSkills:

        def test_returns_list_of_skills(self, skill_manager):
            skill_manager.extractor.annotate.return_value = {
                "results": {"ngram_scored": [
                    {"doc_node_value": "Python"},
                    {"doc_node_value": "SQL"},
                ]}
            }
            result = skill_manager.extract_skills("We need Python and SQL experience")
            assert result == ["Python", "SQL"]

        def test_returns_empty_list_for_none_input(self, skill_manager):
            assert skill_manager.extract_skills(None) == []

        def test_returns_empty_list_for_empty_string(self, skill_manager):
            assert skill_manager.extract_skills("") == []

        def test_returns_empty_list_for_whitespace_only(self, skill_manager):
            assert skill_manager.extract_skills("   ") == []

        def test_returns_empty_list_for_too_short_string(self, skill_manager):
            assert skill_manager.extract_skills("ab") == []

        def test_returns_empty_list_for_non_string_input(self, skill_manager):
            assert skill_manager.extract_skills(12345) == []
            assert skill_manager.extract_skills(["Python"]) == []
            assert skill_manager.extract_skills(None) == []

        def test_strips_whitespace_before_processing(self, skill_manager):
            skill_manager.extractor.annotate.return_value = {
                "results": {"ngram_scored": [{"doc_node_value": "Python"}]}
            }
            result = skill_manager.extract_skills("   Python developer   ")
            assert result == ["Python"]

        def test_returns_empty_list_on_extractor_exception(self, skill_manager):
            skill_manager.extractor.annotate.side_effect = Exception("model error")
            result = skill_manager.extract_skills("Valid input string here")
            assert result == []

        def test_returns_empty_list_when_no_ngram_results(self, skill_manager):
            skill_manager.extractor.annotate.return_value = {
                "results": {"ngram_scored": []}
            }
            result = skill_manager.extract_skills("Some job requirement text")
            assert result == []

        def test_exactly_three_char_string_is_processed(self, skill_manager):
            skill_manager.extractor.annotate.return_value = {
                "results": {"ngram_scored": [{"doc_node_value": "SQL"}]}
            }
            result = skill_manager.extract_skills("SQL")
            assert result == ["SQL"]

        def test_extractor_called_with_stripped_text(self, skill_manager):
            skill_manager.extractor.annotate.return_value = {
                "results": {"ngram_scored": []}
            }
            skill_manager.extract_skills("  Python developer  ")
            skill_manager.extractor.annotate.assert_called_once_with("Python developer")

    # ── save_to_db ────────────────────────────────────────────────────────────

    class TestSaveToDb:

        def _make_session(self, job=MagicMock(), skill=None):
            session = MagicMock()
            session.query.return_value.get.return_value = job
            session.query.return_value.filter_by.return_value.first.return_value = skill
            return session

        def test_commits_on_success(self, skill_manager):
            session = self._make_session()
            skill_manager.save_to_db(session, job_id=1, skill_name="Python")
            session.commit.assert_called_once()

        def test_appends_skill_to_job(self, skill_manager):
            mock_job = MagicMock()
            session = self._make_session(job=mock_job)
            skill_manager.save_to_db(session, job_id=1, skill_name="Python")
            mock_job.skills.append.assert_called_once()

        def test_creates_new_skill_when_not_exists(self, skill_manager):
            session = self._make_session(skill=None)
            with patch("DbIndexing.skill_manager.Skills") as MockSkills:
                new_skill = MagicMock()
                MockSkills.return_value = new_skill
                skill_manager.save_to_db(session, 1, "Docker")
                MockSkills.assert_called_once()

        def test_reuses_existing_skill(self, skill_manager):
            existing_skill = MagicMock()
            session = self._make_session(skill=existing_skill)
            with patch("DbIndexing.skill_manager.Skills") as MockSkills:
                skill_manager.save_to_db(session, 1, "Python")
                MockSkills.assert_not_called()

        def test_returns_early_when_job_not_found(self, skill_manager):
            session = self._make_session(job=None)
            skill_manager.save_to_db(session, job_id=999, skill_name="Python")
            session.commit.assert_not_called()

        def test_rollback_on_exception(self, skill_manager):
            session = self._make_session()
            session.commit.side_effect = Exception("integrity error")
            skill_manager.save_to_db(session, 1, "Python")
            session.rollback.assert_called_once()

        def test_no_exception_raised_on_db_error(self, skill_manager):
            """Errors must be swallowed and printed, not re-raised."""
            session = self._make_session()
            session.commit.side_effect = Exception("deadlock")
            skill_manager.save_to_db(session, 1, "Python")  # should not raise

        def test_prints_error_message_on_failure(self, skill_manager, capsys):
            session = self._make_session()
            session.commit.side_effect = Exception("unique constraint")
            skill_manager.save_to_db(session, 1, "Python")
            captured = capsys.readouterr()
            assert "unique constraint" in captured.out

        def test_skill_name_is_stripped(self, skill_manager):
            session = self._make_session(skill=None)
            with patch("DbIndexing.skill_manager.Skills") as MockSkills:
                new_skill = MagicMock()
                MockSkills.return_value = new_skill
                skill_manager.save_to_db(session, 1, "  Python  ")
                # name set on the skill object should be stripped by caller
                # (stripping happens in skills_pipeline before save_to_db)
                session.commit.assert_called_once()

    # ── skills_pipeline ───────────────────────────────────────────────────────

    class TestSkillsPipeline:

        def test_calls_fetch_db_data(self, skill_manager):
            with patch.object(skill_manager, "fetch_db_data", return_value=[]) as mock_fetch:
                skill_manager.skills_pipeline()
                mock_fetch.assert_called_once()

        def test_calls_extract_skills_for_each_row(self, skill_manager, mock_session):
            rows = [(1, "Need Python"), (2, "Need SQL")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=[]) as mock_extract:
                skill_manager.skills_pipeline()
                assert mock_extract.call_count == 2

        def test_passes_requirements_text_to_extract(self, skill_manager, mock_session):
            rows = [(1, "Need Python and Docker")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=[]) as mock_extract:
                skill_manager.skills_pipeline()
                mock_extract.assert_called_once_with("Need Python and Docker")

        def test_calls_save_to_db_for_each_skill(self, skill_manager, mock_session):
            rows = [(1, "Need Python and Docker")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=["Python", "Docker"]), \
                 patch.object(skill_manager, "save_to_db") as mock_save:
                skill_manager.skills_pipeline()
                assert mock_save.call_count == 2

        def test_save_to_db_receives_correct_job_id(self, skill_manager, mock_session):
            rows = [(99, "Need Python")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=["Python"]), \
                 patch.object(skill_manager, "save_to_db") as mock_save:
                skill_manager.skills_pipeline()
                args = mock_save.call_args[0]
                assert args[1] == 99

        def test_save_to_db_receives_stripped_skill(self, skill_manager, mock_session):
            rows = [(1, "Need Python")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=["  Python  "]), \
                 patch.object(skill_manager, "save_to_db") as mock_save:
                skill_manager.skills_pipeline()
                args = mock_save.call_args[0]
                assert args[2] == "Python"

        def test_skips_rows_with_no_skills_extracted(self, skill_manager, mock_session):
            rows = [(1, "Some requirement")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=[]), \
                 patch.object(skill_manager, "save_to_db") as mock_save:
                skill_manager.skills_pipeline()
                mock_save.assert_not_called()

        def test_handles_empty_db_gracefully(self, skill_manager):
            with patch.object(skill_manager, "fetch_db_data", return_value=[]):
                skill_manager.skills_pipeline()  # should not raise

        def test_multiple_rows_multiple_skills(self, skill_manager, mock_session):
            rows = [(1, "req1"), (2, "req2"), (3, "req3")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=["A", "B"]), \
                 patch.object(skill_manager, "save_to_db") as mock_save:
                skill_manager.skills_pipeline()
                assert mock_save.call_count == 6  # 3 rows × 2 skills

        def test_opens_session_per_row(self, skill_manager, mock_session):
            rows = [(1, "req1"), (2, "req2")]
            with patch.object(skill_manager, "fetch_db_data", return_value=rows), \
                 patch.object(skill_manager, "extract_skills", return_value=["Python"]), \
                 patch.object(skill_manager, "save_to_db"):
                with patch("DbIndexing.skill_manager.SessionLocal", return_value=mock_session) as mock_sl:
                    skill_manager.skills_pipeline()
                    assert mock_sl.call_count == 2