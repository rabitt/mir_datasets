import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--local", action="store_true", default=False, help="run local tests")
    parser.addoption("--dataset", type=str, default="", help="dataset to test locally")
    parser.addoption(
        "--dataset-version",
        type=str,
        default="default",
        help="version of dataset to test locally",
    )
    parser.addoption(
        "--skip-download", action="store_true", default=False, help="skip download step"
    )
    parser.addoption("--report-file", type=str, default="", help="dataset to test locally")


@pytest.fixture(scope="session")
def skip_local(request):
    if request.config.getoption("--local"):
        pytest.skip()


@pytest.fixture(scope="session")
def skip_remote(request):
    if not request.config.getoption("--local"):
        pytest.skip()


@pytest.fixture(scope="session")
def test_dataset(request):
    return request.config.getoption("--dataset")


@pytest.fixture(scope="session")
def dataset_version(request):
    return request.config.getoption("--dataset-version")


@pytest.fixture(scope="session")
def skip_download(request):
    return request.config.getoption("--skip-download")


@pytest.fixture(scope="session")
def report_file(request):
    return request.config.getoption("--report-file")


def pytest_sessionstart(session):
    session.results = dict()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        item.session.results[item] = result


def pytest_sessionfinish(session, exitstatus):
    if len(session.config.option.report_file) > 0:
        report = "\nTests: " + ",".join(session.config.option.file_or_dir) + "\n"
        report += "Dataset: " + str(session.config.option.dataset) + "\n"
        report += "Run status code: " + str(exitstatus) + "\n"
        report += "Running time: " + str(session.config.option.durations_min) + "\n"
        passed_amount = sum(1 for result in session.results.values() if result.passed)
        failed_amount = sum(1 for result in session.results.values() if result.failed)
        report += "There are {} passed and {} failed tests \n".format(passed_amount, failed_amount)
        print(report)
        file_destination = session.config.option.report_file
        if os.path.isdir(os.path.dirname(file_destination)):
            if os.path.exists(file_destination):
                append_write = "a"  # append if already exists
            else:
                append_write = "w"  # make a new file if not

            with open(file_destination, append_write) as txtfile:
                txtfile.write(report + "\n")
        else:
            print("Folder {} does not exist".format(os.path.dirname(file_destination)))
