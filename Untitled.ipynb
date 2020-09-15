{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.options.display.max_columns = None\n",
    "pd.options.display.max_rows = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reason_1</th>\n",
       "      <th>Reason_2</th>\n",
       "      <th>Reaason_3</th>\n",
       "      <th>Reaason_4</th>\n",
       "      <th>Date</th>\n",
       "      <th>Transportation Expense</th>\n",
       "      <th>Distance to Work</th>\n",
       "      <th>Age</th>\n",
       "      <th>Daily Work Load Average</th>\n",
       "      <th>Body Mass Index</th>\n",
       "      <th>Education</th>\n",
       "      <th>Children</th>\n",
       "      <th>Pets</th>\n",
       "      <th>Absenteeism Time in Hours</th>\n",
       "      <th>Month_Value</th>\n",
       "      <th>Day_of_the_week</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2015-07-07</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015-07-14</td>\n",
       "      <td>118</td>\n",
       "      <td>13</td>\n",
       "      <td>50</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2015-07-15</td>\n",
       "      <td>179</td>\n",
       "      <td>51</td>\n",
       "      <td>38</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2015-07-16</td>\n",
       "      <td>279</td>\n",
       "      <td>5</td>\n",
       "      <td>39</td>\n",
       "      <td>239.554</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2015-07-23</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Reason_1  Reason_2  Reaason_3  Reaason_4        Date  \\\n",
       "0         0         0          0          1  2015-07-07   \n",
       "1         0         0          0          0  2015-07-14   \n",
       "2         0         0          0          1  2015-07-15   \n",
       "3         1         0          0          0  2015-07-16   \n",
       "4         0         0          0          1  2015-07-23   \n",
       "\n",
       "   Transportation Expense  Distance to Work  Age  Daily Work Load Average  \\\n",
       "0                     289                36   33                  239.554   \n",
       "1                     118                13   50                  239.554   \n",
       "2                     179                51   38                  239.554   \n",
       "3                     279                 5   39                  239.554   \n",
       "4                     289                36   33                  239.554   \n",
       "\n",
       "   Body Mass Index  Education  Children  Pets  Absenteeism Time in Hours  \\\n",
       "0               30          0         2     1                          4   \n",
       "1               31          0         1     0                          0   \n",
       "2               31          0         0     0                          2   \n",
       "3               24          0         2     0                          4   \n",
       "4               30          0         2     1                          2   \n",
       "\n",
       "   Month_Value  Day_of_the_week  \n",
       "0            7                1  \n",
       "1            7                1  \n",
       "2            7                2  \n",
       "3            7                3  \n",
       "4            7                3  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_preprocessed = pd.read_csv('df_reason_mod.csv')\n",
    "df_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reason_1</th>\n",
       "      <th>Reason_2</th>\n",
       "      <th>Reason_3</th>\n",
       "      <th>Reason_4</th>\n",
       "      <th>Month Value</th>\n",
       "      <th>Day of the Week</th>\n",
       "      <th>Transportation Expense</th>\n",
       "      <th>Distance to Work</th>\n",
       "      <th>Age</th>\n",
       "      <th>Daily Work Load Average</th>\n",
       "      <th>Body Mass Index</th>\n",
       "      <th>Education</th>\n",
       "      <th>Children</th>\n",
       "      <th>Pet</th>\n",
       "      <th>Absenteeism Time in Hours</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>118</td>\n",
       "      <td>13</td>\n",
       "      <td>50</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>179</td>\n",
       "      <td>51</td>\n",
       "      <td>38</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>279</td>\n",
       "      <td>5</td>\n",
       "      <td>39</td>\n",
       "      <td>239.554</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Reason_1  Reason_2  Reason_3  Reason_4  Month Value  Day of the Week  \\\n",
       "0         0         0         0         1            7                1   \n",
       "1         0         0         0         0            7                1   \n",
       "2         0         0         0         1            7                2   \n",
       "3         1         0         0         0            7                3   \n",
       "4         0         0         0         1            7                3   \n",
       "\n",
       "   Transportation Expense  Distance to Work  Age  Daily Work Load Average  \\\n",
       "0                     289                36   33                  239.554   \n",
       "1                     118                13   50                  239.554   \n",
       "2                     179                51   38                  239.554   \n",
       "3                     279                 5   39                  239.554   \n",
       "4                     289                36   33                  239.554   \n",
       "\n",
       "   Body Mass Index  Education  Children  Pet  Absenteeism Time in Hours  \n",
       "0               30          0         2    1                          4  \n",
       "1               31          0         1    0                          0  \n",
       "2               31          0         0    0                          2  \n",
       "3               24          0         2    0                          4  \n",
       "4               30          0         2    1                          2  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_preprocessed = pd.read_csv('C:/Users/abey.assefa/AbeyML/Absenteeism_preprocessed.csv')\n",
    "df_preprocessed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_preprocessed['Absenteeism Time in Hours'].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_preprocessed['Excessive_Absenteeism'] = np.where(df_preprocessed['Absenteeism Time in Hours'] > \n",
    "                                    df_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reason_1</th>\n",
       "      <th>Reason_2</th>\n",
       "      <th>Reason_3</th>\n",
       "      <th>Reason_4</th>\n",
       "      <th>Month Value</th>\n",
       "      <th>Day of the Week</th>\n",
       "      <th>Transportation Expense</th>\n",
       "      <th>Distance to Work</th>\n",
       "      <th>Age</th>\n",
       "      <th>Daily Work Load Average</th>\n",
       "      <th>Body Mass Index</th>\n",
       "      <th>Education</th>\n",
       "      <th>Children</th>\n",
       "      <th>Pet</th>\n",
       "      <th>Absenteeism Time in Hours</th>\n",
       "      <th>Excessive_Absenteeism</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>118</td>\n",
       "      <td>13</td>\n",
       "      <td>50</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>179</td>\n",
       "      <td>51</td>\n",
       "      <td>38</td>\n",
       "      <td>239.554</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>279</td>\n",
       "      <td>5</td>\n",
       "      <td>39</td>\n",
       "      <td>239.554</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>289</td>\n",
       "      <td>36</td>\n",
       "      <td>33</td>\n",
       "      <td>239.554</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Reason_1  Reason_2  Reason_3  Reason_4  Month Value  Day of the Week  \\\n",
       "0         0         0         0         1            7                1   \n",
       "1         0         0         0         0            7                1   \n",
       "2         0         0         0         1            7                2   \n",
       "3         1         0         0         0            7                3   \n",
       "4         0         0         0         1            7                3   \n",
       "\n",
       "   Transportation Expense  Distance to Work  Age  Daily Work Load Average  \\\n",
       "0                     289                36   33                  239.554   \n",
       "1                     118                13   50                  239.554   \n",
       "2                     179                51   38                  239.554   \n",
       "3                     279                 5   39                  239.554   \n",
       "4                     289                36   33                  239.554   \n",
       "\n",
       "   Body Mass Index  Education  Children  Pet  Absenteeism Time in Hours  \\\n",
       "0               30          0         2    1                          4   \n",
       "1               31          0         1    0                          0   \n",
       "2               31          0         0    0                          2   \n",
       "3               24          0         2    0                          4   \n",
       "4               30          0         2    1                          2   \n",
       "\n",
       "   Excessive_Absenteeism  \n",
       "0                      1  \n",
       "1                      0  \n",
       "2                      0  \n",
       "3                      1  \n",
       "4                      0  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.45571428571428574"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## check balace of data\n",
    "df_preprocessed['Excessive_Absenteeism'].sum() / df_preprocessed.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_with_target = df_preprocessed.drop('Absenteeism Time in Hours', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_with_target is df_preprocessed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# input for regression\n",
    "unscaled_inputs = data_with_target.iloc[:, :-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "absenteeism_scale = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "StandardScaler()"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "absenteeism_scale.fit(unscaled_inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaled_inputs = absenteeism_scale.transform(unscaled_inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(700, 14)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaled_inputs.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "## split "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "target = df_preprocessed['Excessive_Absenteeism']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(560, 14) (140, 14) (560,) (140,)\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=20)\n",
    "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
