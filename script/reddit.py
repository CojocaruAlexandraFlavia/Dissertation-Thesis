from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.communication.email import EmailClient
from praw import Reddit
from typing import Dict, Any
from azure.core.exceptions import AzureError
from praw.exceptions import PRAWException
from praw.models import Submission
import sys, os, logging, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script import jira


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%d-%m-%Y %H:%M:%S")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger = logging.getLogger('RedditAnalyzer')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class RedditAnalyzer:
    def __init__(self, client_id, client_secret, password, user_agent, 
                 username, endpoint, key, jira_email, jira_token, base_url, project_id, reporter_id, issue_type):
        self.reddit = Reddit(
            client_id=client_id,
            client_secret=client_secret,
            password=password,
            user_agent=user_agent,
            username=username,
        )
        self.text_analytics_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))
        self.jira_client = jira.JiraClient(jira_email, jira_token, base_url, project_id, reporter_id, issue_type)
        self.email_content = []


    def search_reddits(self, submission_id: str):
        try:
            submission: Submission = self.reddit.submission(submission_id)
            submission.comments.replace_more(limit=None)
            print(f"{len(submission.comments.list())} comments")
            logger.debug(f"{len(submission.comments.list())} comments")
            documents = []
            for index, comment in enumerate(submission.comments.list()):
                try:
                    existing_issue = self.jira_client.search_issue(comment.id)
                    logger.debug(f'Existing issue: {existing_issue}')
                    if len(existing_issue) == 0:
                        documents.append(comment)
                except Exception as e:
                    logger.error(f'Error searching Jira tichet for comment {comment.id}: {e}')
                    self.email_content.extend({
                        "Cause": f"Searching Jira for comment {comment.id}",
                        "Message": f"{e}"
                    })
                    continue 
               
            logger.debug(f"Documents length: {len(documents)}")

            for document in documents:
                self.analyze_comment(document)

            #self.send_email_notification()
        except PRAWException as e:
            logger.error(f'Reddit API error: {e}')
            self.email_content.append({
                "Cause": "Reddit API Error",
                "Message": f"{e}"
            })
            #self.send_email_notification()
            raise(e)
        except AzureError as e:
            logger.error(f'Azure Text Analytics error: {e}')
            self.email_content.append({
                "Cause": "Azure Text Analytics error",
                "Message": f"{e}"
            })
            #self.send_email_notification()
            raise(e)
        except Exception as e:
            logger.error(f'Unexpected error at Reddit data processing: {e}')
            self.email_content.append({
                "Cause": "Unexpected error at Reddit data processing",
                "Message": f"{e}"
            })
            #self.send_email_notification()
            raise(e)


    def analyze_comment(self, comment):
        try:
            biggest_confidence_score = 0
            primary_language = "en"
            language_detection = self.text_analytics_client.detect_language(documents=[comment.body])
            for item in language_detection:
                logger.debug(f"Item: error: {item.is_error}, confidence score: {item.primary_language.confidence_score}, language: {item.primary_language.name}")
                if not item.is_error and item.primary_language.confidence_score > biggest_confidence_score:
                    logger.debug("Confidence score bigger, updating primary language")
                    biggest_confidence_score = item.primary_language.confidence_score
                    primary_language = item.primary_language.iso6391_name
            logger.debug(f"Primary language of the comment in ISO format: {primary_language} with confidence score: {biggest_confidence_score}")
            if not primary_language in ["en", "es", "de", "ko", "ja", "it", "fr", "pt", "zh", "he", "pl"]:
                logger.error(f"Comment language: {primary_language} not support by Azure AI Language service for abstract summarization")
                self.email_content.append({
                    "Cause": "Azure AI Language service for abstract summarization",
                    "Message": f"Comment language: {primary_language} with condifence score {biggest_confidence_score} not support by Azure AI Language service for abstract summarization for comment: {comment.body}"
                })
                return
            logger.debug('Language is supported by service, proceeding with the analysis...')
            analyze_sentiment_result = self.text_analytics_client.analyze_sentiment(documents=[comment.body], show_opinion_mining=True, language=primary_language)
        
            #print(f"Analyze sentiment result length: {len(analyze_sentiment_result)}")
            logger.debug(f"Analyze sentiment result length: {len(analyze_sentiment_result)}")
            for analysis_document in analyze_sentiment_result:
                #print(f'Sentiment: {analysis_document.sentiment}, error: {analysis_document.is_error}')
                logger.debug(f'Sentiment: {analysis_document.sentiment}, error: {analysis_document.is_error}')
                if  not analysis_document.is_error and (analysis_document.sentiment == "negative" or analysis_document.sentiment == "mixed"):
                    summary_message = self.summarize_comment(comment.body, primary_language)               
                    logger.debug(f"summary message: {summary_message}")
                    #print(f"summary message: {summary_message}")

                    target_to_complaints = self.extract_complaints(analysis_document)
                    logger.debug(f"Target_to_complaints: {target_to_complaints}")

                    priority = "High" if analysis_document.sentiment == "negative" else "Medium"

                    for target_name, complaints in target_to_complaints.items():
                        complaint_message =f"{target_name}: " + ", ".join(
                            [assessment.text for complaint in complaints for assessment in complaint.assessments]
                        )
                        logger.debug(f"Complaint message: {complaint_message}")

                        self.jira_client.create_issue(
                            summary=complaint_message.capitalize(),
                            description=f"User have made {len(complaints)} complaint(s) about '{target_name}', specifically saying that it's '{complaint_message}.\n Comment: {comment.id}.\n Summary: {summary_message}'",
                            priority=priority
                        )
        except AzureError as e:
            logger.error(f'Azure Text Analytics error: {e}')
            self.email_content.append({
                "Cause": "Azure Text Analytics error",
                "Message": f"{e}"
            })
            return
        except Exception as e:
            logger.error(f'Error occured at analyzing comment: {e}')
            self.email_content.append({
                "Cause": "Error occured at analyzing comment",
                "Message": f"{e}"
            })
            return


    def summarize_comment(self, comment_body, primary_language):
        try:
            logger.debug(f'Analyzing comment body: {comment_body}')
            abstract_summarization_poller = self.text_analytics_client.begin_abstract_summary(
                documents=[comment_body],
                language=primary_language
            )
            abstract_summarization_result = abstract_summarization_poller.result()
            summary_message = ""
            for item in list(abstract_summarization_result):
                if not item.is_error and len(item.summaries) > 0:
                    for summary in item.summaries:
                        summary_message = summary_message + summary.text + ". "
            return summary_message
        except AzureError as e:
            logger.error(f'Azure Text Analytics error: {e}')
            self.email_content.append({
                "Cause": "Azure Text Analytics error",
                "Message": f"{e}"
            })
            return ""


    def extract_complaints(self, analysis_document):
        target_to_complaints: Dict[str, Any] = {}
        for sentence in analysis_document.sentences:
            if sentence.mined_opinions:
                for mined_opinion in sentence.mined_opinions:
                    target = mined_opinion.target
                    if target.sentiment == 'negative':
                        target_to_complaints.setdefault(target.text, [])
                        target_to_complaints[target.text].append(mined_opinion)
        return target_to_complaints


    def send_email_notification(self):
        try:
            if len(self.email_content) > 0:
                connection_string = os.environ.get('EMAIL_CONNECTION_STRING')
                client = EmailClient.from_connection_string(connection_string)
                recipients = os.environ.get('EMAIL_RECIPIENTS').split(',')

                logger.debug(f'Sending email to: {recipients}')

                message = {
                    "senderAddress": os.environ.get('EMAIL_SENDER'),
                    "recipients":  {
                        "to": [{"address": address } for address in recipients],
                    },
                    "content": {
                        "subject": "Errors found after last pipeline run",
                        "plainText": f"The following issues were raised during the last pipeline execution:\n\n {json.dumps(obj=self.email_content, indent=4)}",
                    }
                }
                client.begin_send(message)
                self.email_content = []
        except Exception as e:
            logger.error(f"Error at sending notification email: {e}")


if __name__ == "__main__":
    
    environment_variables = os.environ
    #print(list(environment_variables.keys()))
    #print(list(environment_variables.items()))
    
    #Reddit environment variables
    client_id = environment_variables.get('REDDIT_CLIENT_ID')
    client_secret = environment_variables.get('REDDIT_CLIENT_SECRET')
    password = environment_variables.get('REDDIT_PASSWORD')
    user_agent = environment_variables.get('REDDIT_USER_AGENT')
    username = environment_variables.get('REDDIT_USERNAME')   
    submission_id = environment_variables.get('REDDIT_SUBMISSION_ID')

    # Azure AI Services environment variables
    endpoint = environment_variables.get('AZURE_AI_ENDPOINT')
    key = environment_variables.get('AZURE_AI_KEY')

    # Jira environment variables
    jira_email = environment_variables.get('JIRA_EMAIL')
    jira_token = environment_variables.get('JIRA_TOKEN')
    base_url = environment_variables.get('JIRA_BASE_URL')
    project_id = environment_variables.get('JIRA_PROJECT_ID')
    reporter_id = environment_variables.get('JIRA_REPORTER_ID')
    issue_type = environment_variables.get('JIRA_ISSUE_TYPE')

    analyzer = RedditAnalyzer(client_id, client_secret, password, user_agent, username, endpoint, key, jira_email, jira_token, 
                              base_url, project_id, reporter_id, issue_type)
    analyzer.search_reddits(submission_id)

