# streamlit_app

## DETAILS STEPS of our streamlit deployment

#### STEP 1: Create repository name `streamlit_app`.
#### STEP 2: Push files needed for deployment. Use the commands below in terminal (MacBook) / Command Prompt (Window):
```python
git init
git add .
git commit -m 'first commit'
git branch -M main
git remote add origin 'HTTPS link'
git push -u origin master
```
* Sometimes, it might need you to enter your **Github Username** and password which is your **Github Personal Access Token**.
```python
Click On Profile Icon > Setting > Developer Setting > Personal Access Tokens > Tokens (Classic)
```
* Then, you can start generate the **Personal Access Token** which can use for 30 days.
  
#### STEP 3: After successfully pushed into Github repository named `streamlit_app`. Go to streamlit website `https://streamlit.io/`.
#### STEP 4: Sign up streamlit account to use streamlit community cloud and Sign into your account.
#### STEP 5: Link streamlit account with Gmail and Github.
#### STEP 6: We can get STARTED to deploy!
#### STEP 7: Click on `New app` and select `From existing repo`.
#### STEP 8: Add `Repository Path`.
#### STEP 9: Branch `main/master` based on your Github main branch.
#### STEP 10: Add the main file path, For example: `streamlit_app.py`. Make sure it is `Python File` with extension `.py`.
#### STEP 11: App URL (Optional), For example: `job-analysis`.streamlit.app (`https://job-analysis.streamlit.app`)
#### STEP 12: Click on `Deploy!`

## We're ready to go! Successfully Deployed!
