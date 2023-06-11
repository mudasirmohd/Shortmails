import traceback2
from flask import Flask
from flask import request, render_template, make_response
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import yaml
import logging

from com.tse.summary_generator import SummaryGenerator

logging.getLogger().setLevel(logging.INFO)

flask_app = Flask(__name__, template_folder='.')
cors = CORS(flask_app)
app = Api(app=flask_app)
name_space = app.namespace('tse', 'The Summarizer Engine')

props = yaml.load(open('prop.yaml'), Loader=yaml.FullLoader)

path = props['path']
D = props['D']
num_clusters = props['num_clusters']

logging.info("Started the app with D  = {}  and  {} number of clusters".format(D, num_clusters))
sum_obj = SummaryGenerator(path + 'df.map', D, num_clusters)


@name_space.route("/")
class SummarizerApi(Resource):
    infer_request = name_space.model("summary_request", {"text": fields.String})

    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)

    @name_space.expect(infer_request)
    def post(self):
        """Takes the text data {'text':''}, runs the summary algorithm,
           produces the summary text result data {'summary':''}

        Returns:
            json -- Result containing the summary
        """
        in_body = request.get_json()
        try:
            text = in_body['text']
            logging.debug(text[:40])
            summary, msg = sum_obj.get_summary_from_text(text)
            email_from = msg['from']
            email_to = msg['to']
            email_date = msg['date']
            email_subject = msg['subject']
            result = {"message": "Success", "summary": summary,
                      'email_from': email_from, 'email_to': email_to, 'email_date': email_date,
                      'email_subject': email_subject}
            logging.debug(result)
        except:
            msg = 'Error in running the TSE:{}'.format(traceback2.format_exc())
            logging.debug(msg)
            # Prepare the result
            result = {"message": msg}
            traceback2.print_exc()

        # Send the result back
        return result


def main():
    logging.info("App Starts...")
    flask_app.run(host='0.0.0.0', port=9092, debug=False)


if __name__ == '__main__':
    main()
