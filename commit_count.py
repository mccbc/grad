import smtplib
import urllib.request
from email.message import EmailMessage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--password',
    type=str,
    help='Gmail password for "From" address')
parser.add_argument(
    '-e',
    '--email',
    action='store_true',
    help='Send an email if there are no new commits')
args = parser.parse_args()

###
your_username = 'mccbc'
###


def send_message(msg):
    msg['Subject'] = '[' + your_username + '] Git Commit Reminder'
    msg['From'] = 'cmcclellan1010@gmail.com'
    msg['To'] = ['cmcclellan1010@gmail.com']

    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login(msg['From'], args.password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string())
    s.quit()


# Scrape HTML from github
with urllib.request.urlopen('https://github.com/' + your_username) as f:
    lines = [line.decode('utf-8') for line in f.readlines()]
    f.close()

# Check the number of commits this month from the scraped HTML
commits = 0
for i, line in enumerate(lines):
    if 'commit' in line and '>' not in line:
        try:
            print('Repository: ', lines[i - 1].split('href="/' + your_username
                   + '/')[1].split('/commits?')[0])
            print(lines[i])
            commits += int(lines[i].lstrip().split(' commits')[0])
        except BaseException:
            pass

# Load running count from last time this script was run
with open('count.txt', 'r') as f:
    previous_commits = int(f.read())  # Will need to reset this every month...
    new_commits = commits - previous_commits

with open('count.txt', 'w') as f:
    f.write(str(commits))

if args.email:
    # Print some statistics
    print('Total commits this month: {}'.format(commits))
    print('Previous commits: {}'.format(previous_commits))
    print('New commits: {}'.format(new_commits))

    msg = EmailMessage()
    if new_commits == 0:
        msg.set_content('You have not committed to any repositories today!')
        send_message(msg)
    elif new_commits < 0:
        msg.set_content('The running total needs to be reset for this month.')
        send_message(msg)
    else:
        pass
