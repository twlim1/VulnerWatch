# Instructions:

## 1. Launch an EC2 instance via gui

Any small instance should be fine. Amazon linux isn't necessary but some commands may need to be reworked on different OSes.

### Add Entries to Security Group
- Add an inbound rule for HTTP so that port 80 is opened up.
- Add an inbound rule for postgresql so that port 5432 is opened up.

Login to the node for the rest of the steps:
`ssh -i <key.pem> ec2-user@<ip>`

## 2. Install initial dependencies

`sudo yum install git`

## 3. Clone project code

### Example 1:
`git clone https://github.com/twlim1/260_capstone.git`

### Example 2, with implied ssh key:
`git clone git@github.com:twlim1/260_capstone.git`

## 4. Run project script
```
cd 260_capstone/src/aws
./containermanage.sh -h
```

### To create fresh images and containers
```
./containermanage.sh fresh
```
- After launching the containers, the database and tables are created but empty. Run the following commands to populate test data
```
./containermanage.sh connect dba
python application/data_download_cve.py --test_mode
python application/batch_prediction.py
```
- To predict severity of cyber security threat from user input description
```
./containermanage.sh connect dba
python application/single_prediction.py --description "A flaw was found in libwebp in versions before ..."
```