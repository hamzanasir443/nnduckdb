import duckdb
import random
import csv


def random_float_array(length):
    return [random.uniform(-1.0, 1.0) for _ in range(length)]


def execute_nn_query_to_csv(connection, atts, limit, lr, iter, parallel, filename='gd_nn.csv'):
    initial_w_xh_query = f"SELECT ARRAY_AGG(wxh) FROM (SELECT wxh_arr[i+1] AS wxh FROM UNNEST(w_xh) AS wxh_arr WHERE w_id = {atts}) AS subquery;"
    initial_w_ho_query = f"SELECT ARRAY_AGG(who) FROM (SELECT who_arr[i+1] AS who FROM UNNEST(w_ho) AS who_arr WHERE w_id = {atts}) AS subquery;"
    
    initial_w_xh = connection.execute(initial_w_xh_query).fetchone()[0]
    initial_w_ho = connection.execute(initial_w_ho_query).fetchone()[0]

    # Create a list to store intermediate results
    intermediate_results = []

    # Perform the recursive updates
    for _ in range(iter):
        # Calculate d_xh and d_ho
        d_xh = []
        d_ho = []
        for i in range(atts):
            d_xh_i = []
            for j in range(4):
                d_xh_i.append(w_xh[i][j] * (1 - w_xh[i][j] * w_xh[i][j]))
            d_xh.append(d_xh_i)

        for i in range(atts):
            d_ho_i = []
            for j in range(3):
                d_ho_i.append(w_ho[i][j] * (1 - w_ho[i][j] * w_ho[i][j]))
            d_ho.append(d_ho_i)

        # Update w_xh and w_ho
        new_w_xh = []
        for i in range(atts):
            new_w_xh_i = []
            for j in range(4):
                new_w_xh_i.append(w_xh[i][j] - lr * sum([img[i] * d for img, d in zip(initial_w_xh, d_xh)]))
            new_w_xh.append(new_w_xh_i)

        new_w_ho = []
        for i in range(atts):
            new_w_ho_i = []
            for j in range(3):
                new_w_ho_i.append(w_ho[i][j] - lr * sum([a_xh[i] * d for a_xh, d in zip(initial_w_ho, d_ho)]))
            new_w_ho.append(new_w_ho_i)

        intermediate_results.append((w_xh, w_ho))
        w_xh = new_w_xh.copy()
        w_ho = new_w_ho.copy()

    # Save intermediate results to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration", "w_xh", "w_ho"])  # headers
        for i, (wxh, who) in enumerate(intermediate_results):
            writer.writerow([i + 1, wxh, who])

# Parameters
parallel = 8
lr = 0.2  # learning rate
attss = [20, 50]
iters = [10, 100, 1000]
limits = [150, 300, 600, 1200]
repeat = 3


# Define Maximum Dimensions
MAX_WXH_ROWS = max(attss)
MAX_WXH_COLS = 4
MAX_WHO_ROWS = max(attss)
MAX_WHO_COLS = 3

# Connect to DuckDB database or create a new one if it doesn't exist
connection = duckdb.connect(database='nn2sql-db', read_only=False)

def flatten_weights(weights, max_rows, max_cols):
    flat = []
    for i in range(max_rows):
        for j in range(max_cols):
            if i < len(weights) and j < len(weights[i]):
                flat.append(weights[i][j])
            else:
                flat.append(None)  # padding with None for missing values
    return flat

try:
    # Begin transaction
    connection.begin()

    # Define a list of table names to be created
    table_names = ["iris", "iris3", "iris2", "img", "one_hot", "w_xh", "w_ho", "weights"]

    # Loop through each table name and check if it exists before creating it
    for table_name in table_names:
        table_check_query = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        result = connection.execute(table_check_query).fetchone()[0]

        if result == 0:
            # Add your SQL code here to create the current table
            sql_code = ""

            if table_name == "iris":
                sql_code = """
                CREATE TABLE iris (sepal_length float, sepal_width float, petal_length float, petal_width float, species int);
                """
            elif table_name == "iris3":
                sql_code = """
                CREATE TABLE iris3 (img float[], one_hot float[]);
                COPY iris FROM './iris.csv' DELIMITER ',' HEADER;
                INSERT INTO iris3 (SELECT ARRAY[sepal_length/10, sepal_width/10, petal_length/10, petal_width/10] AS img,
                                          CASE
                                              WHEN species = 0 THEN ARRAY[1.0, 0.0, 0.0]
                                              WHEN species = 1 THEN ARRAY[0.0, 1.0, 0.0]
                                              ELSE ARRAY[0.0, 0.0, 1.0]
                                          END AS one_hot
                                   FROM iris);
                """
            elif table_name == "iris2":
                sql_code = """
                CREATE TABLE iris2 AS SELECT ROW_NUMBER() OVER () AS id, * FROM iris;
                """
            elif table_name == "img":
                sql_code = """
                CREATE TABLE img AS SELECT id, 1 AS j, sepal_length/10 AS v FROM iris2;
                INSERT INTO img SELECT id, 2, sepal_width/10 FROM iris2;
                INSERT INTO img SELECT id, 3, petal_length/10 FROM iris2;
                INSERT INTO img SELECT id, 4, petal_width/10 FROM iris2;
                """
            elif table_name == "one_hot":
                sql_code = """
                CREATE TABLE one_hot AS
                SELECT n.i, n.j, COALESCE(i.v, 0) AS v, i.v AS dummy
                FROM (SELECT id, species + 1 AS species, 1 AS v FROM iris2) i
                RIGHT OUTER JOIN (SELECT a.a AS i, b.b AS j
                                  FROM (SELECT generate_series AS a FROM generate_series(1, 150)) a,
                                       (SELECT generate_series AS b FROM generate_series(1, 4)) b) n
                ON n.i = i.id AND n.j = i.species
                ORDER BY i, j;
                """
            elif table_name == "w_xh":
                sql_code = """
                CREATE TABLE w_xh (w_id int, i int, j int, v float);
                """
            elif table_name == "w_ho":
                sql_code = """
                CREATE TABLE w_ho (w_id int, i int, j int, v float);
                """
            elif table_name == "weights":
                wxh_columns = ', '.join([f"wxh_{i+1}_{j+1} float" for i in range(MAX_WXH_ROWS) for j in range(MAX_WXH_COLS)])
                who_columns = ', '.join([f"who_{i+1}_{j+1} float" for i in range(MAX_WHO_ROWS) for j in range(MAX_WHO_COLS)])
                sql_code = f"""
                CREATE TABLE weights (
                    wid int,
                    {wxh_columns},
                    {who_columns}
                );
                """
            
            # Execute the SQL code for creating the current table
            connection.execute(sql_code)
            print(f"Table '{table_name}' created.")

        else:
            print(f"Table '{table_name}' already exists.")

    for atts in attss:
        for _ in range(repeat):
            data_for_w_xh = [(atts, i, j, 2 * random.random() - 1) for i in range(1, 5) for j in range(1, atts + 1)]
            data_for_w_ho = [(atts, i, j, 2 * random.random() - 1) for i in range(1, atts + 1) for j in range(1, 4)]

            connection.executemany("INSERT INTO w_xh (w_id, i, j, v) VALUES (?, ?, ?, ?)", data_for_w_xh)
            connection.executemany("INSERT INTO w_ho (w_id, i, j, v) VALUES (?, ?, ?, ?)", data_for_w_ho)

        wxh_arrays = [[random_float_array(4) for _ in range(atts)] for _ in range(4)]
        who_arrays = [[random_float_array(3) for _ in range(atts)] for _ in range(4)]

        insert_columns = ', '.join(['wid'] + [f"wxh_{i+1}_{j+1}" for i in range(MAX_WXH_ROWS) for j in range(MAX_WXH_COLS)]
                                   + [f"who_{i+1}_{j+1}" for i in range(MAX_WHO_ROWS) for j in range(MAX_WHO_COLS)])
        placeholders = ', '.join(['?'] * (1 + MAX_WXH_ROWS * MAX_WXH_COLS + MAX_WHO_ROWS * MAX_WHO_COLS))
        insert_query = f"INSERT INTO weights ({insert_columns}) VALUES ({placeholders})"

        data_for_weights = [(atts, *flatten_weights(wxh, MAX_WXH_ROWS, MAX_WXH_COLS), 
                             *flatten_weights(who, MAX_WHO_ROWS, MAX_WHO_COLS)) for wxh, who in zip(wxh_arrays, who_arrays)]

        connection.executemany(insert_query, data_for_weights)
    for limit in limits:
        for iter in iters:
            for atts in attss:
                execute_nn_query_to_csv(connection, atts, limit, lr, iter, parallel)

    # Commit the transaction
    connection.commit()

    # Inform user about success
    print("Database schema and tables checked/created successfully.")

except Exception as e:
    # Roll back the transaction and print error details
    connection.rollback()
    print("An error occurred:", e)
finally:
    # Close the connection
    connection.close()
