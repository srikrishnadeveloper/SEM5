@echo off
REM Launch JMeter GUI with the autocomplete test plan

cd "C:\Users\srik2\Desktop\College\System Design\tools\apache-jmeter-5.6.3\bin"
start "Apache JMeter" jmeterw.cmd -t "C:\Users\srik2\Desktop\College\System Design\exp7\autocomplete\jmeter-test.jmx"