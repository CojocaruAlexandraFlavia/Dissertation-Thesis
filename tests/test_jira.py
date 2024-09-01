import sys, os, unittest, logging, requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch, MagicMock
from script.jira import JiraClient


class TestJiraClient(unittest.TestCase):
    def setUp(self):
        self.jira_client = JiraClient(
            email="dummy_email",
            token="dummy_token",
            base_url="https://dummy_base_url",
            project_id="dummy_project_id",
            reporter_id="dummy_reporter_id",
            issue_type="dummy_issue_type"
        )

    @patch('script.jira.requests.post')
    def test_create_issue(self, mock_post):
        mock_response = MagicMock()
        mock_response.content = b'{"id": "10000"}'
        mock_post.return_value = mock_response

        self.jira_client.create_issue("Test Summary", "Test Description", "priority")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://dummy_base_url/rest/api/2/issue")
        self.assertIn("summary", kwargs["data"])
        self.assertIn("description", kwargs["data"])
        self.assertIn("dummy_issue_type", kwargs["data"])
        self.assertIn("bugfix", kwargs["data"])
        self.assertIn("dummy_reporter_id", kwargs["data"])


    @patch('script.jira.requests.post')
    def test_create_issue_exception(self, mock_post):
        mock_post.side_effect = requests.RequestException("Test exception")

        with self.assertLogs(level=logging.ERROR) as log:
            with self.assertRaises(Exception) as context:
                self.jira_client.create_issue("summary", "description", "priority")

            self.assertTrue("Test exception" in str(context.exception))
            self.assertTrue(len(log.output) > 0, "No log messages captured")

            expected_log_message = 'ERROR:JiraClient:Error occured at Jira ticket creation: Test exception'
            self.assertIn(expected_log_message, log.output)


    @patch('script.jira.requests.request')
    def test_search_issue(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {'issues': [{'id': '10001'}]}
        mock_post.return_value = mock_response

        issues = self.jira_client.search_issue("dummy_comment_id")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[1], "https://dummy_base_url/rest/api/3/search")
        self.assertEqual(issues, [{'id': '10001'}])


    @patch('script.jira.requests.request')
    def test_search_issue_exception(self, mock_post):
        mock_post.side_effect = requests.RequestException("Test exception")

        with self.assertLogs(level=logging.DEBUG) as log:
            with self.assertRaises(requests.RequestException) as context:
                self.jira_client.search_issue("dummy_comment_id")

            self.assertTrue("Test exception" in str(context.exception))
            self.assertTrue(len(log.output) > 0, "No log messages captured")
            
            expected_log_message = 'ERROR:JiraClient:Error at searching Jira issue for comment ID dummy_comment_id: Test exception'
            self.assertIn(expected_log_message, log.output)


if __name__ == '__main__':
    # pragma: no cover
    unittest.main()