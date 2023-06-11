import yaml
from com.tse.summary_generator import SummaryGenerator
props = yaml.load(open('/Users/mudasir/ShortMail/prop.yaml'), Loader=yaml.FullLoader)

path = props['path']
D = props['D']
num_clusters = props['num_clusters']
text="""First, freeze all of your pip packages in the requirements.txt file using the command

pip freeze > requirements.txt
This should create the requirements.txt file in the correct format. Then try installing using the command

pip install -r requirements.txt
Make sure you're in the same folder as the file when running this command.

If you get some path name instead of the version number in the requirements.txt file, use this pip command to work around it.

pip list --format=freeze > requirements.txt"""
sum_obj = SummaryGenerator(path + 'df.map', D, num_clusters)
summary, msg = sum_obj.get_summary_from_text(text)
file1 = open("MyFile.txt", "w")
print("...................................\n")
file1.write(summary)