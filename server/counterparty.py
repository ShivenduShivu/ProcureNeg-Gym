from copy import deepcopy

from server.models import ActionType, ContractClauses, IPOwnershipType


class Counterparty:
    def __init__(self, reservation: ContractClauses, flexibility: float = 0.1):
        """
        reservation: minimum acceptable contract (hidden)
        flexibility: how much it concedes each round
        """
        self.reservation = deepcopy(reservation)
        self.preferences = {
            "annual_fee": 1.0,
            "payment_terms": 0.8,
            "duration_years": 0.7,
            "sla_uptime": 0.7,
            "sla_penalty_rate": 0.9,
            "liability_cap": 0.9,
            "ip_ownership": 1.0,
            "termination_days": 0.6,
        }
        self.flexibility = flexibility
        self.current_offer = self._build_opening_offer()

    def is_acceptable(self, offer: ContractClauses) -> bool:
        """
        Check if offer meets reservation constraints.
        """
        ip_rank = {
            IPOwnershipType.VENDOR: 0,
            IPOwnershipType.JOINT: 1,
            IPOwnershipType.CLIENT: 2,
        }

        return (
            offer.annual_fee >= self.reservation.annual_fee
            and offer.payment_terms <= self.reservation.payment_terms
            and offer.duration_years >= self.reservation.duration_years
            and offer.sla_uptime <= self.reservation.sla_uptime
            and offer.sla_penalty_rate <= self.reservation.sla_penalty_rate
            and offer.liability_cap <= self.reservation.liability_cap
            and ip_rank[offer.ip_ownership] <= ip_rank[self.reservation.ip_ownership]
            and offer.termination_days >= self.reservation.termination_days
        )

    def generate_counter(self, offer: ContractClauses) -> ContractClauses:
        """
        Move the current counteroffer toward the agent's offer
        without crossing the seller's reservation floor.
        """
        counter = deepcopy(self.current_offer)

        counter.annual_fee = max(
            self.reservation.annual_fee,
            counter.annual_fee - self.flexibility * (counter.annual_fee - offer.annual_fee),
        )

        counter.payment_terms = max(
            15,
            min(
                self.reservation.payment_terms,
                int(
                    counter.payment_terms
                    + self.flexibility * (offer.payment_terms - counter.payment_terms)
                ),
            ),
        )

        counter.duration_years = max(
            self.reservation.duration_years,
            int(
                counter.duration_years
                + self.flexibility * (offer.duration_years - counter.duration_years)
            ),
        )

        counter.sla_uptime = max(
            99.0,
            min(
                self.reservation.sla_uptime,
                counter.sla_uptime
                + self.flexibility * (offer.sla_uptime - counter.sla_uptime),
            ),
        )

        counter.sla_penalty_rate = max(
            self.reservation.sla_penalty_rate,
            counter.sla_penalty_rate
            - self.flexibility * (counter.sla_penalty_rate - offer.sla_penalty_rate),
        )

        counter.liability_cap = max(
            self.reservation.liability_cap,
            counter.liability_cap
            - self.flexibility * (counter.liability_cap - offer.liability_cap),
        )

        counter.ip_ownership = self._counter_ip_ownership(offer.ip_ownership)

        counter.termination_days = max(
            self.reservation.termination_days,
            int(
                counter.termination_days
                + self.flexibility * (offer.termination_days - counter.termination_days)
            ),
        )

        return counter

    def respond(
        self,
        action_type: ActionType,
        offer: ContractClauses,
    ) -> tuple[ActionType, ContractClauses]:
        """
        Returns: (action_type, contract)
        """
        del action_type

        if self.is_acceptable(offer):
            return ActionType.ACCEPT, offer

        counter_offer = self.generate_counter(offer)
        self.current_offer = deepcopy(counter_offer)

        return ActionType.COUNTER, counter_offer

    def _counter_ip_ownership(self, offered_ip: IPOwnershipType) -> IPOwnershipType:
        order = [
            IPOwnershipType.VENDOR,
            IPOwnershipType.JOINT,
            IPOwnershipType.CLIENT,
        ]
        reservation_index = order.index(self.reservation.ip_ownership)
        offered_index = order.index(offered_ip)
        current_index = order.index(self.current_offer.ip_ownership)

        target_index = min(offered_index, reservation_index)
        if target_index == current_index:
            return self.current_offer.ip_ownership

        step_index = current_index + 1 if target_index > current_index else current_index - 1
        step_index = min(reservation_index, max(step_index, target_index))
        return order[step_index]

    def _build_opening_offer(self) -> ContractClauses:
        opening = deepcopy(self.reservation)

        opening.annual_fee = min(2000000, self.reservation.annual_fee * 1.2)
        opening.payment_terms = max(15, int(self.reservation.payment_terms * 0.8))
        opening.duration_years = min(5, self.reservation.duration_years + 1)
        opening.sla_uptime = max(99.0, self.reservation.sla_uptime - 0.05)
        opening.sla_penalty_rate = max(0.01, self.reservation.sla_penalty_rate - 0.02)
        opening.liability_cap = max(0.25, self.reservation.liability_cap - 0.25)
        opening.ip_ownership = IPOwnershipType.VENDOR
        opening.termination_days = min(180, self.reservation.termination_days + 15)

        return opening
