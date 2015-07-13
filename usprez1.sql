insert overwrite local directory 'usprez1'
row format delimited 
fields terminated by "\t"
select id_str, created_at, regexp_replace(text, "[ \t\r\n]+", " "), user.id_str, regexp_replace(user.name, "[ \t\r\n]+", " "), user.screen_name, regexp_replace(user.description, "[ \t\r\n]+", " "), retweeted_status.id_str, retweeted_status.created_at, regexp_replace(retweeted_status.text, "[ \t\r\n]+", " "), retweeted_status.user.id_str, regexp_replace(retweeted_status.user.name, "[ \t\r\n]+", " "), retweeted_status.user.screen_name from gh_rc where year = 2012 and month = 10 and day = 3 and (lower(text) like '%obama%' or lower(text) like '%romney%' or lower(retweeted_status.text) like '%obama%' or lower(retweeted_status.text) like '%romney%');
