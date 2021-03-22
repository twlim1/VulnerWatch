Here's a script that goes through CVE data (not NVD's) and pulls text located at urls from the CVE's 'refs' field. It has 4 supported URLs right now.

- To run the script you'll need to mkdir data in the 'misc' folder. 
- To download data: ```curl -O https://cve.mitre.org/data/downloads/allitems.xml.gz```
- The first run of the script will convert to json and filter out CVEs pre-2020 from ```data/allitems.xml``` and put the resulting subset of CVEs in ```data/someitems.json```
- The script will create 100 files. These files have the url that they come from and the CVE that the respective url came from.
