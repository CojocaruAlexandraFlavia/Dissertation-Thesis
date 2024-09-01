import sys, os, unittest, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch, MagicMock
from unittest import mock, TestCase
from script.reddit import RedditAnalyzer
from parameterized import parameterized
from azure.core.exceptions import AzureError
from praw.exceptions import PRAWException

class TestRedditAnalyzer(TestCase):
    
    @patch('praw.Reddit')
    @patch('azure.ai.textanalytics.TextAnalyticsClient')
    @patch('script.jira.JiraClient')
    @patch('azure.communication.email.EmailClient')
    def setUp(self, mock_email_client, mock_jira_client, mock_text_analytics, mock_reddit):
        self.mock_jira_client = mock_jira_client.return_value
        self.mock_text_analytics_client = mock_text_analytics.return_value
        self.mock_reddit = mock_reddit.return_value
        self.mock_email_client = mock_email_client.return_value
        
        self.reddit_analyzer = RedditAnalyzer(
            client_id='dummy_id',
            client_secret='dummy_secret',
            password='dummy_password',
            user_agent='dummy_agent',
            username='dummy_username',
            endpoint='https://dummy_endpoint',
            key='dummy_key',
            jira_email='dummy_email',
            jira_token='dummy_token',
            base_url='https://dummy_base_url',
            project_id="dummy_project_id",
            reporter_id="dummy_reporter_id",
            issue_type="dummy_issue_type",
            
        )
    

    @patch('script.reddit.RedditAnalyzer.analyze_comment')
    @patch('praw.Reddit.submission')
    @patch('script.jira.JiraClient.search_issue')
    @patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment')
    @patch('azure.communication.email.EmailClient.begin_send')
    def test_search_reddits(self, mock_send_email, mock_analyze_sentiment, mock_search_issue, mock_submission, mock_analyze_comment):
        mock_search_issue.return_value = [
            MagicMock(body='body')
        ]
        mock_comment = MagicMock()
        mock_comment.body = "Test comment body"
        mock_comment.id = "test_comment_id"
        mock_submission_instance = MagicMock()
        mock_submission_instance.comments.list.return_value = [mock_comment]
        mock_submission.return_value = mock_submission_instance

        self.reddit_analyzer.search_reddits('dummy_submission_id')

        mock_submission.assert_called_once_with('dummy_submission_id')


    @patch('script.reddit.RedditAnalyzer.send_email_notification')
    @patch('azure.communication.email.EmailClient.begin_send')
    @patch('praw.Reddit.submission')
    def test_search_reddits_exception_at_search_issue(self, mock_submission, mock_begin_send, mock_send_email):
        
        mock_comment1 = MagicMock()
        mock_comment1.body = "Test comment body"
        mock_comment1.id = "test_comment_id"
        mock_submission_instance = MagicMock()
        mock_submission_instance.comments.list.return_value = [mock_comment1]
        mock_submission.return_value = mock_submission_instance

        self.mock_jira_client.search_issue.side_effect = Exception("Jira search issue failed")

        with self.assertLogs(level='ERROR') as cm:
            self.reddit_analyzer.search_reddits("submission_id")

        self.assertTrue(any("Error searching Jira tichet for comment test_comment_id" in message for message in cm.output))
        self.assertEqual(self.mock_jira_client.search_issue.call_count, 1)


    @parameterized.expand([
        (Exception, 'Unexpected error at Reddit data processing'), 
        (AzureError, 'Azure Text Analytics error'), 
        (PRAWException, 'Reddit API error')
    ])
    @patch("script.reddit.RedditAnalyzer.analyze_comment")
    @patch("script.reddit.Reddit.submission")
    @patch("script.jira.JiraClient.search_issue")
    def test_search_reddits_exception(self, exception, error_message, mock_search_issue, mock_submission, mock_analyze_comment):
        mock_submission.side_effect = exception(error_message)

        with self.assertLogs(level='DEBUG') as log:
            with self.assertRaises(exception) as context:
                self.reddit_analyzer.search_reddits("test_submission_id")
        
        self.assertTrue(error_message in str(context.exception))
        self.assertTrue(any(error_message in message for message in log.output))


    @patch('azure.ai.textanalytics.TextAnalyticsClient.begin_abstract_summary')
    def test_summarize_comment(self, mock_begin_abstract_summary):
        mock_summarization_poller = MagicMock()
        mock_summarization_result = MagicMock()
        mock_summarization_result.is_error = False
        mock_summarization_result.summaries = [MagicMock(text="This is a summary")]
        mock_summarization_poller.result.return_value = [mock_summarization_result]
        mock_begin_abstract_summary.return_value = mock_summarization_poller

        summary = self.reddit_analyzer.summarize_comment('dummy comment body', 'en')

        mock_begin_abstract_summary.assert_called_once_with(documents=['dummy comment body'], language='en')
        self.assertEqual(summary, "This is a summary. ")
    

    @patch('azure.ai.textanalytics.TextAnalyticsClient.begin_abstract_summary')
    def test_summarize_comment_error(self, mock_begin_abstract_summary):
        mock_begin_abstract_summary.side_effect = AzureError('Error')

        with self.assertLogs(level='DEBUG') as log:
            self.reddit_analyzer.summarize_comment("comment body", 'en')
        
        self.assertTrue(any("Azure Text Analytics error:" in message for message in log.output))
        self.assertEqual(mock_begin_abstract_summary.call_count, 1)

    
    @parameterized.expand([
        "negative", "mixed"
    ])
    @patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment')
    @patch('azure.ai.textanalytics.TextAnalyticsClient.detect_language')
    @patch('script.reddit.RedditAnalyzer.summarize_comment')
    def test_analyze_comment(self, sentiment, mock_summarize_comment, mock_detect_language, mock_analyze_sentiment):
        mock_sentiment_analysis = MagicMock()
        mock_sentiment_analysis.sentiment = sentiment
        mock_sentiment_analysis.is_error = False

        mock_sentiment_analysis.sentences = [
            MagicMock(mined_opinions=[
                MagicMock(target=MagicMock(text="Service", sentiment=sentiment), assessments=[
                    MagicMock(text="bad service")
                ])
            ])
        ]
        mock_analyze_sentiment.return_value = [mock_sentiment_analysis]
        mock_summarize_comment.return_value = "This is a summary"

        mock_detect_language.return_value = [MagicMock(
            is_error=False,
            primary_language=MagicMock(
                confidence_score=1.0,
                iso6391_name="en"
            )
        )]

        mock_comment = MagicMock()
        mock_comment.body = "Test comment body"
        mock_comment.id = "test_comment_id"

        self.reddit_analyzer.analyze_comment(mock_comment)

        mock_analyze_sentiment.assert_called_once_with(documents=[mock_comment.body], show_opinion_mining=True, language='en')
        mock_summarize_comment.assert_called_once_with(mock_comment.body, 'en')


    @patch('azure.ai.textanalytics.TextAnalyticsClient.detect_language')
    @patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment')
    @patch('script.reddit.RedditAnalyzer.summarize_comment')
    @patch("script.jira.JiraClient.search_issue")
    @patch('praw.Reddit.submission')
    def test_analyze_comment_error_analyze_sentiment(self, mock_submission, mock_search_issue, mock_summarize_comment, mock_analyze_sentiment, mock_detect_language):
        mock_search_issue.return_value = [
            MagicMock(body='body')
        ]

        mock_comment = MagicMock()
        mock_comment.body = "Test comment body"
        mock_comment.id = "test_comment_id"
        mock_submission_instance = MagicMock()
        mock_submission_instance.comments.list.return_value = [mock_comment]
        mock_submission.return_value = mock_submission_instance
    
        mock_detect_language.side_effect = AzureError("Azure Text Analytics error")

        mock_comment = MagicMock()
        mock_comment.body = "Test comment body"
        mock_comment.id = "test_comment_id"
        with self.assertLogs(level='ERROR') as cm:
            self.reddit_analyzer.analyze_comment(mock_comment)

        self.assertTrue(any("Azure Text Analytics error" in message for message in cm.output))
        self.assertEqual(mock_detect_language.call_count, 1)

    
    @patch('azure.ai.textanalytics.TextAnalyticsClient.detect_language')
    @patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment')
    @patch('script.reddit.RedditAnalyzer.summarize_comment')
    @patch("script.jira.JiraClient.search_issue")
    @patch('praw.Reddit.submission')
    def test_analyze_comment_error_language_not_supported(self, mock_submission, mock_search_issue, mock_summarize_comment, mock_analyze_sentiment, mock_detect_language):
        mock_search_issue.return_value = [
            MagicMock(body='body')
        ]

        mock_comment = MagicMock()
        mock_comment.body = "Test comment body"
        mock_comment.id = "test_comment_id"
        mock_submission_instance = MagicMock()
        mock_submission_instance.comments.list.return_value = [mock_comment]
        mock_submission.return_value = mock_submission_instance
    
        mock_detect_language.return_value = [MagicMock(
            is_error=False,
            primary_language=MagicMock(
                confidence_score=1.0,
                iso6391_name="ro"
            )
        )]

        with self.assertLogs(level='ERROR') as cm:
            self.reddit_analyzer.analyze_comment(mock_comment)

        self.assertTrue(any("Comment language: ro not support by Azure AI Language service for abstract summarization" in message for message in cm.output))
        self.assertEqual(mock_detect_language.call_count, 1)


    @patch('script.reddit.TextAnalyticsClient.detect_language')
    @patch('script.reddit.TextAnalyticsClient.analyze_sentiment')
    @patch('script.reddit.RedditAnalyzer.summarize_comment')
    @patch('script.reddit.RedditAnalyzer.extract_complaints')
    @patch('script.jira.JiraClient.create_issue')
    def test_analyze_comment_error_create_issue(self, mock_create_issue, mock_extract_complaints, mock_summarize_comment, mock_analyze_sentiment, mock_detect_language):
        mock_comment = MagicMock(body='test comment body')
        mock_analysis_document = MagicMock()
        mock_analysis_document.sentiment = 'negative'
        mock_analysis_document.is_error = False
        mock_analysis_document.sentences = []
        mock_analyze_sentiment.return_value = [mock_analysis_document]
        mock_summarize_comment.return_value = 'summary message'
        mock_extract_complaints.return_value = {
            'target_name': [MagicMock(assessments=[MagicMock(text='complaint text')])]
        }
        mock_detect_language.return_value = [MagicMock(
            is_error=False,
            primary_language=MagicMock(
                confidence_score=1.0,
                iso6391_name="en"
            )
        )]

        self.mock_jira_client.create_issue.side_effect = Exception("Jira create issue failed")

        with self.assertLogs(level=logging.ERROR) as cm:
                self.reddit_analyzer.analyze_comment(mock_comment)

        self.assertTrue(any("Jira create issue failed" in message for message in cm.output))
        self.assertEqual(self.mock_jira_client.create_issue.call_count, 1)


    def test_extract_complaints(self):
        mock_sentiment_analysis = MagicMock()
        mock_sentiment_analysis.sentiment = 'negative'
        mock_sentiment_analysis.sentences = [
            MagicMock(mined_opinions=[
                MagicMock(target=MagicMock(text="Service", sentiment='negative'), assessments=[
                    MagicMock(text="bad service")
                ])
            ])
        ]

        target_to_complaints = self.reddit_analyzer.extract_complaints(mock_sentiment_analysis)

        self.assertEqual(len(target_to_complaints), 1)
        self.assertIn("Service", target_to_complaints)
        self.assertEqual(len(target_to_complaints["Service"]), 1)
        self.assertEqual(target_to_complaints["Service"][0].assessments[0].text, "bad service")
    

    @patch('azure.communication.email.EmailClient.from_connection_string')
    @patch('azure.communication.email.EmailClient.begin_send')
    @mock.patch.dict(os.environ, {"EMAIL_RECIPIENTS": "recipients"})
    def test_send_email_notification(self, mock_begin_send, mock_email_client):
        self.reddit_analyzer.email_content = [{"Cause": "cause", "Message": "message"}]
        self.reddit_analyzer.send_email_notification()

        self.assertEqual(len(self.reddit_analyzer.email_content), 0)
    
    
    @patch('azure.communication.email.EmailClient.from_connection_string')
    @patch('azure.communication.email.EmailClient.begin_send')
    def test_send_email_notification_exception(self, mock_begin_send, mock_email_client):
        self.reddit_analyzer.email_content = [{"Cause": "cause", "Message": "message"}]

        with self.assertLogs(level=logging.ERROR) as log:
            self.reddit_analyzer.send_email_notification()

        self.assertTrue(any("Error at sending notification email" in message for message in log.output))


if __name__ == '__main__':
    # pragma: no cover
    unittest.main()