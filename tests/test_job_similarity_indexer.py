import pytest
import numpy as np
import io
from unittest.mock import MagicMock, patch, call, PropertyMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Prevent loading the heavy SentenceTransformer model at import time."""
    with patch("DbIndexing.job_similarity_indexer.MODEL", MagicMock()):
        yield


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def indexer(mock_session, mock_conn):
    with patch("DbIndexing.job_similarity_indexer.SessionLocal", return_value=mock_session), \
         patch("DbIndexing.job_similarity_indexer.engine") as mock_engine:
        mock_engine.connect.return_value = mock_conn
        from DbIndexing.job_similarity_indexer import JobSimilarityIndexer
        yield JobSimilarityIndexer()


def make_row(**kwargs):
    """Return a MagicMock that behaves like a SQLAlchemy mapping row."""
    defaults = {
        "id": 1, "title": "Engineer", "field": "Tech",
        "responsibilities": "Build things", "minimum_requirements": "Python",
        "company": "Acme", "type": "Full-time",
    }
    defaults.update(kwargs)
    row = MagicMock()
    row.get.side_effect = lambda key, default="": defaults.get(key, default)
    return row


def make_embedding(size=384):
    """Return a fake numpy-backed tensor-like object."""
    arr = np.random.rand(size).astype(np.float32)
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.tolist.return_value = arr.tolist()
    return mock_tensor


# ════════════════════════════════════════════════════════════════════════════
# TestJobSimilarityIndexer
# ════════════════════════════════════════════════════════════════════════════

class TestJobSimilarityIndexer:

    # ── Instantiation ─────────────────────────────────────────────────────────

    class TestInstantiation:

        def test_creates_instance(self, indexer):
            from DbIndexing.job_similarity_indexer import JobSimilarityIndexer
            assert isinstance(indexer, JobSimilarityIndexer)

        def test_rows_data_is_none(self, indexer):
            assert indexer.rows_data is None

        def test_df_is_none(self, indexer):
            assert indexer.df is None

        def test_similarity_matrix_is_none(self, indexer):
            assert indexer.similarity_matrix is None

        def test_model_is_assigned(self, indexer):
            assert indexer.model is not None

    # ── determine_offset ──────────────────────────────────────────────────────

    class TestDetermineOffset:

        def test_returns_max_job_id(self, indexer, mock_session):
            mock_session.query.return_value.scalar.return_value = 55
            assert indexer.determine_offset() == 55

        def test_returns_zero_when_table_empty(self, indexer, mock_session):
            mock_session.query.return_value.scalar.return_value = None
            assert indexer.determine_offset() == 0

        def test_returns_integer_type(self, indexer, mock_session):
            mock_session.query.return_value.scalar.return_value = "10"
            result = indexer.determine_offset()
            assert isinstance(result, int)

        def test_uses_session_context_manager(self, indexer, mock_session):
            mock_session.query.return_value.scalar.return_value = 0
            indexer.determine_offset()
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

        def test_session_error_propagates(self, indexer, mock_session):
            mock_session.query.side_effect = Exception("DB down")
            with pytest.raises(Exception, match="DB down"):
                indexer.determine_offset()

    # ── retrieve_jobs_data ────────────────────────────────────────────────────

    class TestRetrieveJobsData:

        def test_populates_rows_data(self, indexer, mock_session, mock_conn):
            mock_session.query.return_value.scalar.return_value = 0
            fake_rows = [make_row(id=1), make_row(id=2)]
            mock_conn.execute.return_value.mappings.return_value.all.return_value = fake_rows
            indexer.retrieve_jobs_data()
            assert indexer.rows_data == fake_rows

        def test_rows_data_empty_when_no_jobs(self, indexer, mock_session, mock_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            indexer.retrieve_jobs_data()
            assert indexer.rows_data == []

        def test_uses_engine_connect(self, indexer, mock_session, mock_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            indexer.retrieve_jobs_data()
            mock_conn.__enter__.assert_called_once()

        def test_applies_offset(self, indexer, mock_session, mock_conn):
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            with patch.object(indexer, "determine_offset", return_value=20) as mock_offset:
                indexer.retrieve_jobs_data()
                mock_offset.assert_called_once()

        def test_engine_error_propagates(self, indexer, mock_session, mock_conn):
            mock_session.query.return_value.scalar.return_value = 0
            mock_conn.execute.side_effect = Exception("timeout")
            with pytest.raises(Exception, match="timeout"):
                indexer.retrieve_jobs_data()

    # ── encode_job ────────────────────────────────────────────────────────────

    class TestEncodeJob:

        def test_returns_embedding_and_job_id(self, indexer):
            row = make_row(id=7)
            fake_embedding = make_embedding()
            indexer.model.encode.return_value = fake_embedding
            embedding, job_id = indexer.encode_job(row)
            assert job_id == 7
            assert embedding is fake_embedding

        def test_job_id_is_integer(self, indexer):
            row = make_row(id="42")
            indexer.model.encode.return_value = make_embedding()
            _, job_id = indexer.encode_job(row)
            assert isinstance(job_id, int)
            assert job_id == 42

        def test_combined_text_includes_all_fields(self, indexer):
            row = make_row(
                title="Dev", field="IT", responsibilities="Code",
                minimum_requirements="Python", company="Corp", type="Remote"
            )
            indexer.model.encode.return_value = make_embedding()
            indexer.encode_job(row)
            call_text = indexer.model.encode.call_args[0][0]
            assert "Dev" in call_text
            assert "IT" in call_text
            assert "Code" in call_text
            assert "Python" in call_text
            assert "Corp" in call_text
            assert "Remote" in call_text

        def test_missing_fields_default_to_empty_string(self, indexer):
            row = MagicMock()
            row.get.return_value = ""
            indexer.model.encode.return_value = make_embedding()
            embedding, job_id = indexer.encode_job(row)
            assert job_id == 0

        def test_encode_called_with_tensor_flag(self, indexer):
            row = make_row()
            indexer.model.encode.return_value = make_embedding()
            indexer.encode_job(row)
            _, kwargs = indexer.model.encode.call_args
            assert kwargs.get("convert_to_tensor") is True

        def test_model_error_propagates(self, indexer):
            row = make_row()
            indexer.model.encode.side_effect = RuntimeError("OOM")
            with pytest.raises(RuntimeError, match="OOM"):
                indexer.encode_job(row)

    # ── batch_encode_all_jobs ─────────────────────────────────────────────────

    class TestBatchEncodeAllJobs:

        def test_returns_none_when_rows_data_is_none(self, indexer):
            indexer.rows_data = None
            assert indexer.batch_encode_all_jobs() is None

        def test_returns_none_when_rows_data_is_empty(self, indexer):
            indexer.rows_data = []
            assert indexer.batch_encode_all_jobs() is None

        def test_returns_list_of_embeddings(self, indexer):
            fake_emb = make_embedding()
            indexer.rows_data = [make_row(id=1), make_row(id=2)]
            with patch.object(indexer, "encode_job", return_value=(fake_emb, 1)), \
                 patch.object(indexer, "store_embedding"):
                result = indexer.batch_encode_all_jobs()
                assert isinstance(result, list)
                assert len(result) == 2

        def test_calls_encode_job_for_each_row(self, indexer):
            rows = [make_row(id=i) for i in range(3)]
            indexer.rows_data = rows
            fake_emb = make_embedding()
            with patch.object(indexer, "encode_job", return_value=(fake_emb, 1)) as mock_enc, \
                 patch.object(indexer, "store_embedding"):
                indexer.batch_encode_all_jobs()
                assert mock_enc.call_count == 3

        def test_calls_store_embedding_for_each_row(self, indexer):
            rows = [make_row(id=i) for i in range(3)]
            indexer.rows_data = rows
            fake_emb = make_embedding()
            with patch.object(indexer, "encode_job", return_value=(fake_emb, 1)), \
                 patch.object(indexer, "store_embedding") as mock_store:
                indexer.batch_encode_all_jobs()
                assert mock_store.call_count == 3

        def test_returns_early_when_encode_returns_none_embedding(self, indexer):
            indexer.rows_data = [make_row(id=1), make_row(id=2)]
            with patch.object(indexer, "encode_job", return_value=(None, 1)), \
                 patch.object(indexer, "store_embedding") as mock_store:
                result = indexer.batch_encode_all_jobs()
                assert result is None
                mock_store.assert_not_called()

        def test_returns_early_when_encode_returns_none_job_id(self, indexer):
            indexer.rows_data = [make_row(id=1)]
            fake_emb = make_embedding()
            with patch.object(indexer, "encode_job", return_value=(fake_emb, None)), \
                 patch.object(indexer, "store_embedding") as mock_store:
                result = indexer.batch_encode_all_jobs()
                assert result is None
                mock_store.assert_not_called()

        def test_continues_on_store_embedding_exception(self, indexer):
            rows = [make_row(id=1), make_row(id=2)]
            indexer.rows_data = rows
            fake_emb = make_embedding()
            with patch.object(indexer, "encode_job", return_value=(fake_emb, 1)), \
                 patch.object(indexer, "store_embedding", side_effect=Exception("DB error")):
                result = indexer.batch_encode_all_jobs()
                # continues past store errors but embeddings list stays empty
                assert result == []

    # ── store_embedding ───────────────────────────────────────────────────────

    class TestStoreEmbedding:

        def test_adds_and_commits(self, indexer, mock_session):
            indexer.store_embedding(1, [0.1, 0.2, 0.3])
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

        def test_creates_job_embeddings_object(self, indexer, mock_session):
            with patch("DbIndexing.job_similarity_indexer.JobEmbeddings") as MockEmb:
                indexer.store_embedding(42, [0.1])
                MockEmb.assert_called_once_with(job_id=42, embedding=[0.1])

        def test_swallows_exception_and_prints(self, indexer, mock_session, capsys):
            mock_session.add.side_effect = Exception("constraint violation")
            indexer.store_embedding(1, [0.1])  # must not raise
            captured = capsys.readouterr()
            assert "constraint violation" in captured.out

        def test_uses_session_context_manager(self, indexer, mock_session):
            indexer.store_embedding(1, [0.1])
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

    # ── load_embeddings ───────────────────────────────────────────────────────

    class TestLoadEmbeddings:

        def test_returns_list_of_numpy_arrays(self, indexer, mock_conn):
            fake_rows = [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
            mock_conn.execute.return_value.mappings.return_value.all.return_value = fake_rows
            result = indexer.load_embeddings()
            assert len(result) == 2
            assert all(isinstance(e, np.ndarray) for e in result)

        def test_returns_empty_list_when_no_rows(self, indexer, mock_conn):
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            result = indexer.load_embeddings()
            assert result == []

        def test_uses_engine_connect(self, indexer, mock_conn):
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            indexer.load_embeddings()
            mock_conn.__enter__.assert_called_once()

        def test_engine_error_propagates(self, indexer, mock_conn):
            mock_conn.execute.side_effect = Exception("read error")
            with pytest.raises(Exception, match="read error"):
                indexer.load_embeddings()

    # ── store_sim_matrix ──────────────────────────────────────────────────────

    class TestStoreSimMatrix:

        def test_adds_and_commits(self, indexer, mock_session):
            sim_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            indexer.store_sim_matrix(sim_matrix)
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

        def test_serialises_matrix_as_bytes(self, indexer, mock_session):
            sim_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
            with patch("DbIndexing.job_similarity_indexer.JobSimilarityMatrix") as MockJSM:
                indexer.store_sim_matrix(sim_matrix)
                _, kwargs = MockJSM.call_args
                assert isinstance(kwargs["matrix"], bytes)

        def test_bytes_roundtrip_preserves_matrix(self, indexer, mock_session):
            original = np.array([[1.0, 0.3], [0.3, 1.0]])
            captured = {}

            def capture(**kwargs):
                captured["matrix"] = kwargs["matrix"]
                return MagicMock()

            with patch("DbIndexing.job_similarity_indexer.JobSimilarityMatrix", side_effect=capture):
                indexer.store_sim_matrix(original)

            buf = io.BytesIO(captured["matrix"])
            buf.seek(0)
            recovered = np.load(buf)
            np.testing.assert_array_almost_equal(original, recovered)

        def test_uses_session_context_manager(self, indexer, mock_session):
            indexer.store_sim_matrix(np.array([[1.0]]))
            mock_session.__enter__.assert_called_once()
            mock_session.__exit__.assert_called_once()

        def test_session_error_propagates(self, indexer, mock_session):
            mock_session.add.side_effect = Exception("write error")
            with pytest.raises(Exception, match="write error"):
                indexer.store_sim_matrix(np.array([[1.0]]))

    # ── load_sim_matrix ───────────────────────────────────────────────────────

    class TestLoadSimMatrix:

        def _make_matrix_bytes(self, matrix: np.ndarray) -> bytes:
            buf = io.BytesIO()
            np.save(buf, matrix)
            buf.seek(0)
            return buf.read()

        def test_loads_and_sets_similarity_matrix(self, indexer, mock_session):
            original = np.array([[1.0, 0.4], [0.4, 1.0]])
            mock_obj = MagicMock()
            mock_obj.matrix = self._make_matrix_bytes(original)
            mock_session.query.return_value.order_by.return_value.first.return_value = mock_obj
            indexer.load_sim_matrix()
            np.testing.assert_array_almost_equal(indexer.similarity_matrix, original)

        def test_does_nothing_when_no_matrix_in_db(self, indexer, mock_session):
            mock_session.query.return_value.order_by.return_value.first.return_value = None
            indexer.load_sim_matrix()
            assert indexer.similarity_matrix is None

        def test_uses_most_recent_entry(self, indexer, mock_session):
            original = np.array([[1.0]])
            mock_obj = MagicMock()
            mock_obj.matrix = self._make_matrix_bytes(original)
            order_by_mock = mock_session.query.return_value.order_by.return_value
            order_by_mock.first.return_value = mock_obj
            indexer.load_sim_matrix()
            mock_session.query.return_value.order_by.assert_called_once()

        def test_uses_session_context_manager(self, indexer, mock_session):
            mock_session.query.return_value.order_by.return_value.first.return_value = None
            indexer.load_sim_matrix()
            mock_session.__enter__.assert_called_once()

    # ── calculate_sim_matrix ──────────────────────────────────────────────────

    class TestCalculateSimMatrix:

        def _rand_embeddings(self, n=3, size=8):
            return [np.random.rand(size).astype(np.float32) for _ in range(n)]

        def test_returns_none_when_both_none(self, indexer):
            with patch.object(indexer, "load_embeddings", return_value=None), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=None):
                result = indexer.calculate_sim_matrix()
                assert result is None

        def test_uses_only_new_when_old_is_none(self, indexer):
            new_embs = self._rand_embeddings()
            with patch.object(indexer, "load_embeddings", return_value=None), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=new_embs), \
                 patch("DbIndexing.job_similarity_indexer.cosine_similarity") as mock_cos:
                mock_cos.return_value = np.array([[1.0]])
                indexer.calculate_sim_matrix()
                mock_cos.assert_called_once_with(new_embs)

        def test_uses_only_old_when_new_is_none(self, indexer):
            old_embs = self._rand_embeddings()
            with patch.object(indexer, "load_embeddings", return_value=old_embs), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=None), \
                 patch("DbIndexing.job_similarity_indexer.cosine_similarity") as mock_cos:
                mock_cos.return_value = np.array([[1.0]])
                indexer.calculate_sim_matrix()
                mock_cos.assert_called_once_with(old_embs)

        def test_uses_both_when_both_present(self, indexer):
            old_embs = self._rand_embeddings(2)
            new_embs = self._rand_embeddings(3)
            with patch.object(indexer, "load_embeddings", return_value=old_embs), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=new_embs), \
                 patch("DbIndexing.job_similarity_indexer.cosine_similarity") as mock_cos:
                mock_cos.return_value = np.ones((3, 2))
                indexer.calculate_sim_matrix()
                mock_cos.assert_called_once_with(new_embs, old_embs)

        def test_returns_cosine_similarity_result(self, indexer):
            embs = self._rand_embeddings()
            expected = np.eye(3)
            with patch.object(indexer, "load_embeddings", return_value=None), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=embs), \
                 patch("DbIndexing.job_similarity_indexer.cosine_similarity", return_value=expected):
                result = indexer.calculate_sim_matrix()
                np.testing.assert_array_equal(result, expected)

        def test_load_embeddings_called(self, indexer):
            with patch.object(indexer, "load_embeddings", return_value=None) as mock_load, \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=None):
                indexer.calculate_sim_matrix()
                mock_load.assert_called_once()

        def test_batch_encode_called(self, indexer):
            with patch.object(indexer, "load_embeddings", return_value=None), \
                 patch.object(indexer, "batch_encode_all_jobs", return_value=None) as mock_enc:
                indexer.calculate_sim_matrix()
                mock_enc.assert_called_once()

    # ── Integration: full pipeline ────────────────────────────────────────────

    class TestFullPipeline:

        def test_pipeline_runs_without_error(self, indexer, mock_session, mock_conn):
            sim_matrix = np.eye(2)
            mock_conn.execute.return_value.mappings.return_value.all.return_value = []
            mock_session.query.return_value.scalar.return_value = 0

            with patch.object(indexer, "retrieve_jobs_data"), \
                 patch.object(indexer, "calculate_sim_matrix", return_value=sim_matrix), \
                 patch.object(indexer, "store_sim_matrix") as mock_store:
                indexer.retrieve_jobs_data()
                result = indexer.calculate_sim_matrix()
                indexer.store_sim_matrix(result)
                mock_store.assert_called_once_with(sim_matrix)

        def test_store_not_called_when_matrix_is_none(self, indexer):
            with patch.object(indexer, "retrieve_jobs_data"), \
                 patch.object(indexer, "calculate_sim_matrix", return_value=None), \
                 patch.object(indexer, "store_sim_matrix") as mock_store:
                indexer.retrieve_jobs_data()
                result = indexer.calculate_sim_matrix()
                if result is not None:
                    indexer.store_sim_matrix(result)
                mock_store.assert_not_called()