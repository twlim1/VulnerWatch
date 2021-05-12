# Instructions:

## 1. Launch an EC2 instance via gui

Any small instance should be fine. Amazon linux isn't necessary but some commands may need to be reworked on different OSes.

It is necessary to add an entry to the automatically created security group. Add an inbound rule for HTTP so that port 80 is opened up.

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


