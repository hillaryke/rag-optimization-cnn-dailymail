import unittest
import pandas as pd
from langchain.docstore.document import Document

import os
os.chdir("../../")
from src.rag_pipeline.load_docs import load_docs_from_csv  # Import your function

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
        result = load_docs_from_csv(
            file_path=None,  # Here we use the DataFrame directly for this test
            page_content_column="article",
            as_document=True,
        )

if __name__ == "__main__":
    unittest.main()
