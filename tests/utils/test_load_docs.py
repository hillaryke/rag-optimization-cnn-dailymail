import unittest
import pandas as pd
from langchain.docstore.document import Document

import os
os.chdir("../../")
from src.rag_pipeline.load_docs import load_docs_from_csv

from unittest.mock import patch  # patch for mocking

class TestLoadDocsFromCSV(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame (replace with actual data if needed)
        self.test_data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "article": [
                    "This is article 1.",
                    "Another interesting article 2 here.",
                    "The final article, number 3.",
                ],
            }
        )

    def test_load_as_documents(self):
        """Test loading documents as Document objects with metadata."""
        # Mock the pd.read_csv to return your test data directly
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = self.test_data
            result = load_docs_from_csv(
                file_path="test_data/test_file.csv",  # Here I Provide a valid path for mocking
                page_content_column="article",
                as_document=True,
            )

        # Check if the result is a list of Document objects
        self.assertIsInstance(result, list)
        for doc in result:
            self.assertIsInstance(doc, Document)
            # Check if the metadata is correct
            self.assertEqual(doc.metadata["source"], "cnn_dailymail")
            self.assertIn(doc.metadata["id"], self.test_data["id"].tolist())


if __name__ == "__main__":
    unittest.main()
