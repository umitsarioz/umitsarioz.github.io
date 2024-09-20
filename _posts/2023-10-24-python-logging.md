---
date: 2023-10-24
title: Logging instead of Print Statements!
image: /assets/img/ss/2023-10-24-logging/hello_world.png
#categories: [Software]
tags: [python, software,tricks, logging]
published: true
math: true
description: Print statements are handy for logging and debugging, but they have some limitations and risks.
---

Many of us started writing code by printing a hello world. **Print statements are useful in many places such as logging and debugging while writing code, but they also have some limits.** Print statements print an output to the console screen at the end of the day. Depending on the system used and default stdout buffer size settings, your code will unintentionally throw an error when the output in the console reaches the **maximum stdout buffer size.** In this case, while you are looking for the cause of the error in the general functioning of your code, you will be very surprised when you see that it actually comes from the print statement. In addition, reviewing an output with **long print statements** is a very difficult task, and as time goes by, you will realize that it makes you very tired. In summary, although print statements make our job much easier in small projects and rapid development, **we must be very careful about the places we use when we go to productization.** To deal with this situation, you can use some logging tools or modules written to keep logs.

<hr>

Logging is used to record while debugging or to keep track of some steps in the code at run time.There are many helpful tools for this. In this post, I will briefly talk about the simple but useful <a href="https://docs.python.org/3/howto/logging.html">Python logging module</a> and how it can be used.
A logging object is simply defined within the logging library. Within the definitions, you can define information such as writing to the file, the mode in which you will write to the file, the log format it will print, and the log levels you want.** By connecting your code snippets or projects that run at certain times to the logging library, you can regularly save the log information to files and review it when necessary.** This feature allows us to develop safely without worrying about the print buffer size problem, helps us examine the logs of our codes over time and what actions we can take.

> **TL;DR:**  Logging gives us three main benefits: reliability, flexibility, easy and fast debugging.
{: .prompt-info}

<hr>

Let's assume we code a simple project with capabilities like the following:
- Connecting to the database,
- Performing some operations on the connected database,
- Measuring the code duration and issuing a warning when it exceeds a certain time,
- Throwing an error when a required file cannot be found in the code,
- Closing the database connection

When we write the codes, design the project, and include the logging library in this project, we expect to get an output like Figure 1.

![demo.png](/assets/img/ss/2023-10-24-logging/demo.png)
_Figure 1. Logging output example of the codes are below._

> **Note:** ls and cat are linux commands. ls command list files in current directory, cat command shows content of a file.
{: .prompt-info}

<hr>

Let's quickly pseudo-code the project we have assumed and examine how we include and use the logging library in it.
``` python
import logging
import time
from datetime import datetime

def connect_db():
    # connect db some connection parameters and package
    time.sleep(1)  # sleep 1 second
    logging.info("Connected to database")

def do_some_cool_things():
    # do some cool things like get a table and select some operations on rows
    time.sleep(1)  # sleep 1 second
    logging.info("Cool things is done!")

def be_angry():
    # let's assume code raise an FileNotFound exception
    time.sleep(1)  # sleep 1 second
    logging.error("FileNotFoundError exist!")

def give_an_advice():
    # let's assume code runs for too long
    time.sleep(1)  # sleep 1 second
    logging.warning("The code has been working for a long time.")

def close_db_connection():
    # close database connection if not closed before
    time.sleep(1)  # sleep 1 second
    logging.info("Database connection is closed.")

def main():
    logging.basicConfig(format='%(asctime)s :: [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filename=f'db_pipeline_{datetime.now().strftime("%Y%m%d")}.log',
                        filemode='w',
                        encoding='utf-8')
    connect_db()
    do_some_cool_things()
    give_an_advice()
    be_angry()
    close_db_connection()

if __name__ == "__main__":
    main()

```

<hr>

>"In summary, although print statements make our work much easier in small projects and rapid development, we should be very careful about the places we use when we go to productization. Using alternative modules such as logging instead of print statements makes our job easier, gives us flexibility, and ensures that our code runs reliably and allows us to monitor it."
{: .prompt-tip}

Thanks for reading.