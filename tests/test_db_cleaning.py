import pytest
from unittest.mock import MagicMock, patch, call, PropertyMock
from datetime import datetime, timedelta
from sqlalchemy import delete, select


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session():
    """Provides a MagicMock session that also works as a context manager."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


@pytest.fixture
def db_cleaner(mock_session):
    """Returns a DbCleaner with SessionLocal patched to our mock session."""
    with patch("DbIndexing.db_cleaning.SessionLocal", return_value=mock_session):
        from DbIndexing.db_cleaning import DbCleaner
        yield DbCleaner()


@pytest.fixture
def fixed_now():
    return datetime(2024, 6, 15, 12, 0, 0)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_executed_stmt(mock_session, call_index=0):
    """Return the first positional argument of the nth execute() call."""
    return mock_session.execute.call_args_list[call_index][0][0]


# ════════════════════════════════════════════════════════════════════════════
# TestDbCleaner
# ════════════════════════════════════════════════════════════════════════════

class TestDbCleaner:

    # ── Instantiation ────────────────────────────────────────────────────────

    def test_instantiation(self):
        from DbIndexing.db_cleaning import DbCleaner
        cleaner = DbCleaner()
        assert isinstance(cleaner, DbCleaner)

    # ── delete_expired_jobs ──────────────────────────────────────────────────

    class TestDeleteExpiredJobs:

        def test_executes_delete_statement(self, db_cleaner, mock_session):
            db_cleaner.delete_expired_jobs()
            mock_session.execute.assert_called_once()

        def test_commits_after_delete(self, db_cleaner, mock_session):
            db_cleaner.delete_expired_jobs()
            mock_session.commit.assert_called_once()

        def test_commit_follows_execute(self, db_cleaner, mock_session):
            call_order = []
            mock_session.execute.side_effect = lambda *a, **kw: call_order.append("execute")
            mock_session.commit.side_effect = lambda: call_order.append("commit")
            db_cleaner.delete_expired_jobs()
            assert call_order == ["execute", "commit"]

        @patch("DbIndexing.db_cleaning.datetime")
        def test_uses_current_datetime(self, mock_dt, db_cleaner, mock_session, fixed_now):
            mock_dt.now.return_value = fixed_now
            db_cleaner.delete_expired_jobs()
            mock_dt.now.assert_called_once()

        def test_uses_session_context_manager(self, db_cleaner, mock_session):
            db_cleaner.delete_expired_jobs()
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

        def test_does_not_rollback_on_success(self, db_cleaner, mock_session):
            db_cleaner.delete_expired_jobs()
            mock_session.rollback.assert_not_called()

        def test_session_error_propagates(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = Exception("DB error")
            with pytest.raises(Exception, match="DB error"):
                db_cleaner.delete_expired_jobs()

    # ── delete_low_quality_jobs ──────────────────────────────────────────────

    class TestDeleteLowQualityJobs:

        def test_is_a_no_op(self, db_cleaner, mock_session):
            """Current implementation is `pass`; must not touch the session."""
            db_cleaner.delete_low_quality_jobs()
            mock_session.execute.assert_not_called()
            mock_session.commit.assert_not_called()

        def test_returns_none(self, db_cleaner):
            assert db_cleaner.delete_low_quality_jobs() is None

        def test_does_not_raise(self, db_cleaner):
            db_cleaner.delete_low_quality_jobs()  # no exception expected

    # ── delete_old_job_postings ──────────────────────────────────────────────

    class TestDeleteOldJobPostings:

        def test_executes_delete_statement(self, db_cleaner, mock_session):
            db_cleaner.delete_old_job_postings()
            mock_session.execute.assert_called_once()

        def test_commits_after_delete(self, db_cleaner, mock_session):
            db_cleaner.delete_old_job_postings()
            mock_session.commit.assert_called_once()

        def test_commit_follows_execute(self, db_cleaner, mock_session):
            call_order = []
            mock_session.execute.side_effect = lambda *a, **kw: call_order.append("execute")
            mock_session.commit.side_effect = lambda: call_order.append("commit")
            db_cleaner.delete_old_job_postings()
            assert call_order == ["execute", "commit"]

        @patch("DbIndexing.db_cleaning.datetime")
        def test_cutoff_is_45_days_ago(self, mock_dt, db_cleaner, mock_session, fixed_now):
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            db_cleaner.delete_old_job_postings()
            mock_dt.now.assert_called_once()

        def test_uses_session_context_manager(self, db_cleaner, mock_session):
            db_cleaner.delete_old_job_postings()
            mock_session.__enter__.assert_called_once()

        def test_session_error_propagates(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = RuntimeError("timeout")
            with pytest.raises(RuntimeError, match="timeout"):
                db_cleaner.delete_old_job_postings()

    # ── delete_old_similarity_matrix ─────────────────────────────────────────

    class TestDeleteOldSimilarityMatrix:

        def test_executes_delete_statement(self, db_cleaner, mock_session):
            db_cleaner.delete_old_similarity_matrix()
            mock_session.execute.assert_called_once()

        def test_commits_after_delete(self, db_cleaner, mock_session):
            db_cleaner.delete_old_similarity_matrix()
            mock_session.commit.assert_called_once()

        def test_commit_follows_execute(self, db_cleaner, mock_session):
            call_order = []
            mock_session.execute.side_effect = lambda *a, **kw: call_order.append("execute")
            mock_session.commit.side_effect = lambda: call_order.append("commit")
            db_cleaner.delete_old_similarity_matrix()
            assert call_order == ["execute", "commit"]

        @patch("DbIndexing.db_cleaning.datetime")
        def test_cutoff_is_7_days_ago(self, mock_dt, db_cleaner, mock_session, fixed_now):
            mock_dt.now.return_value = fixed_now
            db_cleaner.delete_old_similarity_matrix()
            mock_dt.now.assert_called_once()

        def test_session_error_propagates(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = Exception("connection reset")
            with pytest.raises(Exception, match="connection reset"):
                db_cleaner.delete_old_similarity_matrix()

    # ── delete_duplicate_jobs ────────────────────────────────────────────────

    class TestDeleteDuplicateJobs:

        def test_executes_delete_statement(self, db_cleaner, mock_session):
            db_cleaner.delete_duplicate_jobs()
            mock_session.execute.assert_called_once()

        def test_commits_after_delete(self, db_cleaner, mock_session):
            db_cleaner.delete_duplicate_jobs()
            mock_session.commit.assert_called_once()

        def test_commit_follows_execute(self, db_cleaner, mock_session):
            call_order = []
            mock_session.execute.side_effect = lambda *a, **kw: call_order.append("execute")
            mock_session.commit.side_effect = lambda: call_order.append("commit")
            db_cleaner.delete_duplicate_jobs()
            assert call_order == ["execute", "commit"]

        def test_uses_session_context_manager(self, db_cleaner, mock_session):
            db_cleaner.delete_duplicate_jobs()
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

        def test_session_error_propagates(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = Exception("constraint error")
            with pytest.raises(Exception, match="constraint error"):
                db_cleaner.delete_duplicate_jobs()

        def test_execute_called_exactly_once(self, db_cleaner, mock_session):
            """Duplicate jobs need a single composite delete, not multiple passes."""
            db_cleaner.delete_duplicate_jobs()
            assert mock_session.execute.call_count == 1

    # ── delete_orphaned_relationships ────────────────────────────────────────

    class TestDeleteOrphanedRelationships:

        def test_executes_three_delete_statements(self, db_cleaner, mock_session):
            db_cleaner.delete_orphaned_relationships()
            assert mock_session.execute.call_count == 3

        def test_commits_three_times(self, db_cleaner, mock_session):
            db_cleaner.delete_orphaned_relationships()
            assert mock_session.commit.call_count == 3

        def test_each_execute_is_followed_by_commit(self, db_cleaner, mock_session):
            call_order = []
            mock_session.execute.side_effect = lambda *a, **kw: call_order.append("execute")
            mock_session.commit.side_effect = lambda: call_order.append("commit")
            db_cleaner.delete_orphaned_relationships()
            assert call_order == [
                "execute", "commit",
                "execute", "commit",
                "execute", "commit",
            ]

        def test_uses_session_context_manager(self, db_cleaner, mock_session):
            db_cleaner.delete_orphaned_relationships()
            mock_session.__enter__.assert_called_once()

        def test_first_failure_aborts_remaining_deletes(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = Exception("FK violation")
            with pytest.raises(Exception, match="FK violation"):
                db_cleaner.delete_orphaned_relationships()
            assert mock_session.execute.call_count == 1

        def test_second_failure_stops_third_delete(self, db_cleaner, mock_session):
            mock_session.execute.side_effect = [None, Exception("FK violation")]
            with pytest.raises(Exception, match="FK violation"):
                db_cleaner.delete_orphaned_relationships()
            assert mock_session.execute.call_count == 2

    # ── Integration: full clean pipeline ─────────────────────────────────────

    class TestFullCleanPipeline:

        def test_all_methods_callable_in_sequence(self, db_cleaner, mock_session):
            """Smoke-test the __main__ execution order without errors."""
            db_cleaner.delete_old_job_postings()
            db_cleaner.delete_expired_jobs()
            db_cleaner.delete_duplicate_jobs()
            db_cleaner.delete_old_similarity_matrix()
            db_cleaner.delete_orphaned_relationships()

        def test_total_execute_calls_across_pipeline(self, db_cleaner, mock_session):
            # old_postings=1, expired=1, duplicates=1, similarity=1, orphans=3
            db_cleaner.delete_old_job_postings()
            db_cleaner.delete_expired_jobs()
            db_cleaner.delete_duplicate_jobs()
            db_cleaner.delete_old_similarity_matrix()
            db_cleaner.delete_orphaned_relationships()
            assert mock_session.execute.call_count == 7

        def test_total_commit_calls_across_pipeline(self, db_cleaner, mock_session):
            db_cleaner.delete_old_job_postings()
            db_cleaner.delete_expired_jobs()
            db_cleaner.delete_duplicate_jobs()
            db_cleaner.delete_old_similarity_matrix()
            db_cleaner.delete_orphaned_relationships()
            assert mock_session.commit.call_count == 7

        def test_low_quality_noop_does_not_add_calls(self, db_cleaner, mock_session):
            db_cleaner.delete_low_quality_jobs()
            assert mock_session.execute.call_count == 0
            assert mock_session.commit.call_count == 0