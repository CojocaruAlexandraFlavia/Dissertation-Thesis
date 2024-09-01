import requests, json, logging, sys
from requests.auth import HTTPBasicAuth

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%d-%m-%Y %H:%M:%S")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger = logging.getLogger('JiraClient')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class JiraClient:
    def __init__(self, email, token, base_url, project_id, reporter_id, issue_type):
        self.email = email
        self.token = token
        self.base_url = base_url
        self.project_id = project_id
        self.reporter_id = reporter_id
        self.issue_type = issue_type

    def create_issue(self, summary: str, description: str, priority: str):
        try:
            url = f"{self.base_url}/rest/api/2/issue"
            logger.debug(f'Jira URL: {url}')
            auth = HTTPBasicAuth(self.email, self.token)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            payload = json.dumps({
                "fields": {
                    "description": description,
                    "issuetype": {
                        "id": f"{self.issue_type}"
                    },
                    "labels": [
                        "bugfix",
                    ],
                    "project": {
                        "id": f"{self.project_id}"
                    },
                    "reporter": {
                        "id": f"{self.reporter_id}"
                    },
                    "summary": summary,
                    "priority": {
                        "name": priority
                    }
                }
            })
            logger.debug(payload)

            response = requests.post(url, data=payload, headers=headers, auth=auth)
            response.raise_for_status()
            #print(response.content)
        except requests.RequestException as e:
            logger.error(f'Error occured at Jira ticket creation: {e}')
            raise(e)
        #    print(f"Error: {e}")

    def search_issue(self, comment_id):
        try:
            url = f"{self.base_url}/rest/api/3/search"
            logger.debug(f'Jira URL: {url}')
            auth = HTTPBasicAuth(self.email, self.token)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            query = {
                'jql': f'description ~ {comment_id}',
                'maxResults': 1,
                "expand": [
                    "names",
                    "schema",
                ],
                "fields": [
                    "summary"
                ],
            }

            logger.debug(f'Request payload: {json.dumps(obj=query, indent=4)}')
            response = requests.request(
                "GET",
                url,
                headers=headers,
                params=query,
                auth=auth
            )
            logger.debug(f'Search Jira issue response: {json.dumps(obj=response.json(), indent=4)}')
            response.raise_for_status()
            return response.json().get('issues', [])
        except requests.RequestException as e:
            logger.error(f'Error at searching Jira issue for comment ID {comment_id}: {e}')
            raise(e)
