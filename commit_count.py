import smtplib
import urllib.request
from email.message import EmailMessage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--password', type=str, help='Gmail password for "From" address')
args = parser.parse_args()

###
your_username = 'mccbc'
###

with urllib.request.urlopen('https://github.com/'+your_username) as f:
    lines = [line.decode('utf-8') for line in f.readlines()]
    f.close()

commits_today = 0 
for i, line in enumerate(lines): 
    if 'commit' in line and '>' not in line: 
        try: 
            print('Repository: ', lines[i-1].split('href="/'+your_username+'/')[1].split('/commits?')[0]) 
            print(lines[i]) 
            commits_today += int(lines[i].lstrip()[0])
        except: 
            pass 

print('Total commits today: {}'.format(commits_today))

if commits_today == 0:
    msg = EmailMessage()
    msg.set_content('You have not committed to any repositories today!')
    msg['Subject'] = '['+your_username+'] Git Commit Reminder'
    msg['From'] = 'cmcclellan1010@gmail.com'
    msg['To'] = ['cmcclellan1010@gmail.com']

    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login(msg['From'], args.password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string())
    s.quit()
