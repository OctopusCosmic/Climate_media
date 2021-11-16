From William Wu..

This is just a small snippet of the bill data I am extracting so you can see all the connections between files easily.
From the bill_frame CSV, I have a connection to all the other CSV's via the bill_id. If the bill_id matches one of the rows 
in the other tables, then it is correlated to that specific id. I'm planning on transferring the full data CSV's into mysql 
tables later on, although that might change in the future.