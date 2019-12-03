from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QWidget


class StepProgressBar(QWidget):
    def __init__(self):
        super(StepProgressBar, self).__init__()
        self.layout = QHBoxLayout()
        self.value = -1
        self.steps = []
        self.current_step_index = 0

        # Analyze image
        self.analyze_step = Step("Analyze")
        self.analyze_step.setToolTip("Analyzing input")
        self.layout.addWidget(self.analyze_step)
        self.steps.append(self.analyze_step)

        # Prepare the model
        self.prepare_step = Step("Prepare the model")
        self.prepare_step.setToolTip("Preparing the model")
        self.layout.addWidget(self.prepare_step)
        self.steps.append(self.prepare_step)

        # Train the model
        self.train_step = Step("Train the model")
        self.train_step.setToolTip("Analyzing input")
        self.layout.addWidget(self.train_step)
        self.steps.append(self.train_step)

        # Infer and Save the result
        self.infersave_step = Step("Predict and Save")
        self.infersave_step.setToolTip("Analyzing input")
        self.layout.addWidget(self.infersave_step)
        self.steps.append(self.infersave_step)

        self.setLayout(self.layout)
        self.steps[0].setStyleSheet("background-color: orange")

    def emit(self, val):
        if val is not self.value:
            print("emit called")
            if self.current_step_index < len(self.steps):
                self.steps[self.current_step_index].setStyleSheet(
                    "background-color: green"
                )
            self.current_step_index += 1
            if self.current_step_index < len(self.steps):
                self.steps[self.current_step_index].setStyleSheet(
                    "background-color: orange"
                )
            self.value = val


class Step(QPushButton):
    def mousePressEvent(self, QMouseEvent):
        return
